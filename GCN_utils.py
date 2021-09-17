'''
Utilities used in graph network creation and training. 

**List of Functions**
- gnn_model_summary: Prints model params and layers 
- get_graph: Creates a graph of nodes/edges 
- GCNSeq (Class): The data generator class (specifically, for 2-signal types used in GCN study). 
  - Other datagens exist but this one is the "final" one (hopefully can upload more if time permits) 
- test: A function for testing the model performance (compute loss, accuracy, return those scores) 
- get_gcn_predictions: return the predictions for classification (probabilistics) for a datagen 
'''
from tqdm import tqdm 

import torch
import pickle

import networkx as nx
import awkward as ak
from torch_geometric.utils import from_networkx, add_self_loops
from torch_geometric.data import Data
from data.data_preprocessing import load_pmt_positions
import numpy as np
import tensorflow as tf 

def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

def get_graph(max_dist=700., min_dist=200.0, 
              add_weights=False, medium='wbls', 
              loadpath=None, return_nx=False,
):
    '''
    get a torch graph of the PMT info. 
    
    max_dist: Maximum distance for PMT relation (euclidian)
    min_dist: Minimum distance (<=) for PMT relation (use -1 to include Loops)
    
    add_weights: If True, add inverse weighted distance.
    medium: either "wbls" or "h2o"
    '''
    print('***Creating Graph...***')
    pmtxyz = load_pmt_positions(medium=medium)
    n_pmts = len(pmtxyz)
    if loadpath is not None: 
        G = nx.read_gpickle(loadpath)
        torch_graph = from_networkx(G)
        torch_graph.pos = pmtxyz
        return torch_graph
    # else manually calculate the graph 
    pmtx, pmty, pmtz = pmtxyz[:,0], pmtxyz[:,1], pmtxyz[:,2]
    dist = np.zeros((n_pmts, n_pmts))
    for i, pmtpos in enumerate(pmtxyz):
        dist[i] = np.sqrt(np.sum((pmtxyz - pmtxyz[i])**2, axis=1))
    all_edges = np.argwhere(dist<max_dist)
    all_weights = 1. - dist/np.max(dist)
    loops = np.argwhere(dist<=min_dist)
    all_weights = 1. - dist/np.max(dist)
    sel_weights = all_weights[np.logical_and(dist!=0, dist<max_dist)]
    all_edges = np.argwhere(np.logical_and(dist!=0, dist<max_dist))
    
    G = nx.Graph()
    if return_nx: return G
    if add_weights: G.add_weighted_edges_from(edges_and_weights)
    else: G.add_edges_from(all_edges)
    G.remove_edges_from(loops)

    torch_graph = from_networkx(G)
    torch_graph.pos = pmtxyz
    print('***Graph Created...***')
    return torch_graph 
    
def get_dist_matrix(medium='wbls', 
                    metric='euclidian',
):
    '''
    get a distance matrix of the PMT info. 

    medium: either "wbls" or "h2o"
    metric: 'euclidian' is the only supported one for now
    '''
    pmtxyz = load_pmt_positions(medium=medium)
    n_pmts = len(pmtxyz)

    pmtx, pmty, pmtz = pmtxyz[:,0], pmtxyz[:,1], pmtxyz[:,2]
    dist = np.zeros((n_pmts, n_pmts))
    for i, pmtpos in enumerate(pmtxyz):
        dist[i] = np.sqrt(np.sum((pmtxyz - pmtxyz[i])**2, axis=1))
    return dist 

class GCNSeq(tf.keras.utils.Sequence):
    def __init__(self, filename, torch_graph, batch_size=64, shuffle=True, ):
        data = pickle.load(open("./data/train_test_sets/%s.pkl"%(filename), 'rb'))
        self.x, self.x2, self.y = data['ev1'], data['ev2'], ak.to_numpy(data['y'])
        self.x.newcharge = self.x.pmtcharge
        self.x.newcharge = ak.where(self.x.hittime<950, self.x.pmtcharge, 0)
        self.x.newcharge = ak.where(self.x.hittime>750, self.x.newcharge, 0)
        self.batch_size = batch_size
        self.pmt_pos = np.expand_dims(torch_graph.pos, axis=0)
        self.n_pmts = len(torch_graph.pos)
        self.pmtpos_iter = np.repeat(self.pmt_pos, self.batch_size, axis=0)
        self.pmtpos_iter = self.pmtpos_iter/np.max(self.pmtpos_iter, axis=1, keepdims=True)
        self.n_samples = len(self.y)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.torch_graph=torch_graph
    def __len__(self):
        return np.int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):      
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size     
        
        indexes = self.indexes[i0:i1]        
        batch_y = np.expand_dims(self.y[indexes], axis=1)

        dat = self.x[indexes]
        charge = np.zeros((len(batch_y), self.n_pmts))
        times = np.zeros_like(charge)
        for i in range(len(batch_y)):
            charge[i, dat.channel[i]]= charge[i, dat.channel[i]] + dat.pmtcharge[i]
            times[i, dat.channel[i]]= (dat.hittime[i]-300.)/1500.
        charge = np.reshape( charge, (len( charge ), self.n_pmts,1) )
        times = np.reshape( times, (len( times ), self.n_pmts,1) )
        x = np.concatenate( (charge, self.pmtpos_iter[:len(batch_y)]), axis=-1).astype(np.float32)
        x = np.concatenate( (x, times), axis=-1).astype(np.float32)
        
        proc_data = Data(x=torch.tensor(x), edge_index=self.torch_graph.edge_index, 
                                y=torch.tensor(batch_y, dtype=torch.float), pos=self.torch_graph.pos)            
        return proc_data
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def test(model, datagen, criterion, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    totsize = len(datagen.y)
    batch_size=datagen.batch_size
    with torch.no_grad():
        with tqdm(datagen, unit="batch") as tepoch:
            tepoch.set_description(f'Validating...')
            for i, data in enumerate(tepoch):
                inputs = data.to(device)
                labels = inputs.y
                outputs = model(inputs)
                pred = torch.round(torch.sigmoid(outputs))# outputs.round()
                correct += (pred == labels).sum().item()
                loss += criterion(outputs, labels).detach().item()
                tepoch.set_postfix(loss=loss/(i+1), accuracy=correct/((i+1)*batch_size))
        accuracy = correct/(totsize)
        return accuracy, loss/(i+1)

# def plot_distance_matrix():
def get_gcn_predictions(model, datagen, device='cpu'):
    model.eval()
    correct = 0.0
    loss = 0.0
    totsize = len(datagen.y)
    net_out = torch.zeros((len(datagen.y),1))
    batch_size=datagen.batch_size
    with torch.no_grad():
        with tqdm(datagen, unit="batch") as tepoch:
            tepoch.set_description(f'Validating..')
            for i, data in enumerate(tepoch):
                inputs = data.to(device)
                outputs = torch.sigmoid(model(inputs).detach())
                net_out[i*batch_size:(i+1)*batch_size] = outputs
        del inputs, outputs
        return net_out