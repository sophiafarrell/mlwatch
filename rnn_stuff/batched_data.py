import pickle 
import numpy as np 
import awkward as ak
import tensorflow as tf 

class DatSequence(tf.keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, tres=5, shuffle=True):
        data = pickle.load(open("./data/%s.pkl"%(filename), 'rb'))
        self.x1, self.x2, self.y = data['ev1'], data['ev2'], data['y']
        self.batch_size = batch_size
        self.n_samples = len(self.y)
        self.npmts = 2330
        self.timetot = 1500 #ns 
        self.tres = tres # ns 
        self.nbins = int(self.timetot/self.tres)    
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return np.int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):      
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size     
        
        indexes = self.indexes[i0:i1]        
        batch_y = self.y[indexes]

        first = np.zeros((len(batch_y), self.npmts, self.nbins), dtype=np.float32)
        second = np.zeros_like(first)
        
        signal1, signal2 = self.x1[indexes], self.x2[indexes]

        bins1 = ak.values_astype((signal1.hittime-300)/self.tres, np.int32)
        bins2 = ak.values_astype((signal2.hittime-300)/self.tres, np.int32)
        
        for x, data, bins in zip([first, second], 
                                 [signal1, signal2], 
                                 [bins1, bins2]):
            for i in range(len(batch_y)):
                x[i, data.channel[i], bins[i]]=x[i, data.channel[i], bins[i]]+data.pmtcharge[i]
        
        return [first, second], np.asarray(batch_y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)