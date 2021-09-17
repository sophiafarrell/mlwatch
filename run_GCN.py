import torch
from tqdm import tqdm

from torch_geometric.nn import GCNConv, ASAPooling, TopKPooling
from torch.nn import Linear, Sequential, ReLU, Dropout, Sigmoid, Tanh

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.set_device(1)
device = torch.device("cuda")

from data.data_preprocessing import *
from GCN_utils import *

class GCNNet(torch.nn.Module):
    def __init__(self, final_out=3, n_pmts=2330):
        super(GCNNet, self).__init__()
        self.lin = Sequential(Linear(n_pmts*final_out, 32),
#                               Dropout(0.1),
                              nn.BatchNorm1d(32),
                              ReLU(), 
#                               Linear(32, 32),
#                               ReLU(), 
                              Linear(32, 1),
#                               Tanh(), 
                             )
        self.conv1 = GCNConv(5, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, final_out)
#         self.pool1 = TopKPooling(in_channels=n_pmts, ratio=0.5)
        self.final_out=final_out
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
#         edge_weight = data.edge_weight
        x = self.conv1(x, edge_index,)
        x = F.relu(x)
        x = self.conv2(x, edge_index,)
        x = F.relu(x)
        x = self.conv3(x, edge_index,)
        x = F.relu(x)
        x = x.view(-1, self.final_out*n_pmts)
        x = self.lin(x)
        return x
    

tgkw = dict(medium='h2o', max_dist=600.0, min_dist=0.,)
torch_graph = get_graph(**tgkw # max is 1584 cm 
)
n_pmts = len(torch_graph.pos)
model = GCNNet(n_pmts=n_pmts).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3,)
gnn_model_summary(model)


def train(train_data, test_data=None, patience=5, max_epochs=30):
    train_accuracies, test_accuracies = list(), list()
    train_losses, test_losses = list(), list()
    batch_size=train_data.batch_size
    previous_test_acc = 0.0

    for epoch in range(1, max_epochs+1):
        running_loss = 0.0
        tot_loss=0.0
        accuracy=0.0
        tot_right=0.0
        i=1
        model.train()
        with tqdm(train_data, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}/{max_epochs}")
            for data in tepoch:
                inputs = data.to(device)
                labels = inputs.y
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                predictions = torch.round(torch.sigmoid(outputs)) # outputs.round()
                tot_right += (predictions == labels).sum().item()
                accuracy = tot_right/(i*batch_size)
                tot_loss += loss.item()
                running_loss = tot_loss/i
                i+=1
                
                tepoch.set_postfix(loss=running_loss, accuracy=100. * accuracy)
        train_acc, train_loss = accuracy, running_loss
        test_acc, test_loss = test(model, test_data, criterion, device)
        if test_acc>=previous_test_acc:
            torch.save(model.state_dict(), './GCN_models/model_water_last.pkl')
            patience=patience
            previous_test_acc = test_acc
        else: patience-=1
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print('Epoch: {:03d}/{:03d}, Loss: {:.5f}, Val. Loss: {:.5f}, Train Acc: {:.5f}, Val. Acc: {:.5f}'.
              format(epoch, max_epochs, train_loss, test_loss, train_acc, test_acc))
        if patience<=0:
            print('***Reached maximum patience level in training. Exiting early***')
            return
    return

# datagen = GCNSeq('pn_train', batch_size=10)
# val_datagen = GCNSeq('pn_test', batch_size=10)
medium=tgkw['medium']
medium = 'water'
datagen = GCNSeq(f'train_{medium}_2sig', torch_graph=torch_graph, batch_size=16)
val_datagen = GCNSeq(f'test_{medium}_2sig', torch_graph=torch_graph, batch_size=16)
train(datagen, val_datagen, max_epochs=30)
