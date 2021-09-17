import pickle 
import numpy as np 
import awkward as ak
import tensorflow as tf 

class DatSequence(tf.keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, shuffle=True):
        data = pickle.load(open("./data/%s.pkl"%(filename), 'rb'))
        self.x1, self.x2, self.y = data['ev1'], data['ev2'], data['y']
        self.batch_size = batch_size
        self.n_samples = len(self.y)
        self.npmts = 2330
        self.shuffle = shuffle
        self.on_epoch_end()
        self.out_shape = (63, 37)
    def __len__(self):
        return np.int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):      
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size     
        
        indexes = self.indexes[i0:i1]        
        batch_y = self.y[indexes]

        first = np.zeros((len(batch_y), self.npmts+1, 1), dtype=np.float32)
        second = np.zeros_like(first)        
        signal1, signal2 = self.x1[indexes], self.x2[indexes]

        
        for x, data in zip([first, second], 
                                 [signal1, signal2], 
                          ):
            data.newcharge = data.pmtcharge
            data.newcharge = ak.where(data.hittime<950, data.pmtcharge, 0)
            data.newcharge = ak.where(data.hittime>750, data.newcharge, 0)
            masker = data.newcharge>0       
            
            for i in range(len(batch_y)):
                x[i, data.channel[i], 0]=x[i, data.channel[i], 0]+data.pmtcharge[i]
            #reshape to something square. Just try it out. 
            x = x.reshape(len(batch_y), self.out_shape[0], self.out_shape[1])
            
        return [first, second], np.asarray(batch_y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
class EmbeddingSeq(tf.keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, max_length=200, shuffle=True):
        data = pickle.load(open("./data/%s.pkl"%(filename), 'rb'))
        self.x1, self.x2, self.y = data['ev1'], data['ev2'], data['y']
        self.batch_size = batch_size
        self.n_samples = len(self.y)
        self.max_length=max_length
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return np.int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):      
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size     
        
        indexes = self.indexes[i0:i1]        
        batch_y = self.y[indexes]
    
        signal1, signal2 = self.x1[indexes], self.x2[indexes]
        to_return = []
        for i, data in enumerate([signal1,signal2]):
            data.newcharge = data.pmtcharge
            data.newcharge = ak.where(data.hittime<950, data.pmtcharge, 0)
            data.newcharge = ak.where(data.hittime>750, data.newcharge, 0)
            masker = data.newcharge>0       
            x = tf.keras.preprocessing.sequence.pad_sequences(data.channel[masker], maxlen=self.max_length)
            to_return.append(x)
        return to_return, np.asarray(batch_y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
class PNSequence(tf.keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, max_length=200, shuffle=True):
        data = pickle.load(open("./data/%s.pkl"%(filename), 'rb'))
        self.x, self.y = data['x'], data['y']
        self.batch_size = batch_size
        self.n_samples = len(self.y)
        self.max_length=max_length
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return np.int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):      
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size     
        
        indexes = self.indexes[i0:i1]        
        batch_y = self.y[indexes]
    
        signal = self.x[indexes]
        signal.newcharge = signal.pmtcharge
        signal.newcharge = ak.where(signal.hittime<950, signal.pmtcharge, 0)
        signal.newcharge = ak.where(signal.hittime>750, signal.newcharge, 0)
        masker = signal.newcharge>0       
        batch_x = tf.keras.preprocessing.sequence.pad_sequences(signal.channel[masker], maxlen=self.max_length)
        return batch_x, np.asarray(batch_y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)