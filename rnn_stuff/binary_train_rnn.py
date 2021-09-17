import uproot
import numpy as np
import pandas as pd
import awkward as ak
from scipy.stats import norm
import scipy.interpolate as interpolate
from scipy.ndimage import median as med

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Input, InputLayer
from tensorflow.keras.layers import Concatenate, Flatten, Bidirectional, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks, regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        gpu = gpus[1]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        print(e)

#Setup and preprocess data 
pmtpos = np.loadtxt('pmtpositions.csv', delimiter=',') + 1.
pmtpos = np.vstack((np.array([0, 0, 0]), pmtpos))

fastn = ak.from_json('data/fastn.json')
ibd = ak.from_json('data/ibd.json')

for j, data in enumerate([fastn, ibd]):
    args = ak.argsort(data['restime'])
    for key in ['hittime', 'pmtcharge', 'channel', 'restime']:
        data[key] = data[key][args]


def prep_net_data(data_to_manipulate, 
                  prompt=True, restime=True,
                  maxlen=100):
    X = []
    y = np.array([])
    
    if prompt==True:
        keys = ['promptchan', 'promptcharge', 'prompttime']
    elif prompt==False:
        keys = ['channel', 'pmtcharge', 'hittime']
    if restime==True:
        keys.append('restime')
        
    diminput = len(keys)
    for d, dat in enumerate(data_to_manipulate):
        print('Running set %i of %i'%(d+1, len(data_to_manipulate)))
        new = np.empty(shape=(len(dat), maxlen, diminput+2))

        chn = sequence.pad_sequences(dat[keys[0]]+1, maxlen=maxlen, 
                                     padding='post', 
                                     dtype='int32')
        chrg = sequence.pad_sequences(dat[keys[1]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
        
        xyz = pmtpos[chn] #since value of 0 has meaning..
        
        time = sequence.pad_sequences(dat[keys[2]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
        new[:,:,0] = chrg
        new[:,:,1] = time    
        new[:,:,2:5] = xyz
        
        if restime==True:
            res_times = sequence.pad_sequences(dat[keys[3]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
            new[:,:,5] = res_times
            
        if len(X)>0:
            X = np.append(X, new, axis=0)
        else: 
            X = new
        y=np.append(y, d*np.ones(len(new)))    
    if np.max(y)>1:
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
    return X, y

all_data = ak.concatenate([ibd, fastn])

fn = all_data['code']==2
invbeta = all_data['code']==1
id0 = all_data['subid']==0
id1p = all_data['subid']>0 # >0 changes things 

ibd0_mask = ak.from_iter([all(t) for t in zip(invbeta, id0)])
ibd1_mask = ak.from_iter([all(t) for t in zip(invbeta, id1p)])

samples = min([ak.count_nonzero(fn), 
              ak.count_nonzero(invbeta), 
              ])

# x1 = all_data[fn][:samples]
# x1 = ak.concatenate((x1, all_data[ibd0_mask][:samples]))
# x1 = ak.concatenate((x1, all_data[ibd1_mask][:samples]))

data_to_manipulate = [all_data[fn][:samples],
                      all_data[invbeta][:samples], 
                     ]

# X1 = get_dims(x1, dimensions=dimensions)
# X1 = ak.to_pandas(X1)
X2, y = prep_net_data(data_to_manipulate, prompt=False, restime=True, maxlen=200)

# Split dataset into training set and test setd
X_train2, X_test2, y_train, y_test = train_test_split(X2, y, test_size=0.25, random_state=43) 

## MODEL BUILDING 
regressor2 = Sequential()
input_shape = X2.shape[1:]
regressor2.add(InputLayer(input_shape=input_shape))
regressor2.add(LSTM(units=256, 
                    return_sequences=True, 
                    dropout=0.2,
                  ))
regressor2.add(LSTM(units=128, 
                   return_sequences=True, 
#                    dropout=0.2,
                  ))
regressor2.add(LSTM(units=64, 
                   return_sequences=True, 
                   dropout=0.2, 
                  ))
regressor2.add(LSTM(units=32, 
                    return_sequences=True, 
                    dropout=0.2, 
                   ))
# regressor2.add(LSTM(units=16, 
#                     return_sequences=True, 
#                     dropout=0.2, 
#                    ))

regressor2.add(Flatten())

regressor2.add(Dense(256))
regressor2.add(LeakyReLU(alpha=0.05))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(128))
regressor2.add(LeakyReLU(alpha=0.05))
# regressor2.add(Dropout(0.2))

regressor2.add(Dense(64))
regressor2.add(LeakyReLU(alpha=0.05))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(32))
regressor2.add(LeakyReLU(alpha=0.05))

regressor2.add(Dense(1, activation='sigmoid'))

regressor2.compile(loss='binary_crossentropy', 
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                 metrics=['accuracy']
                )
regressor2.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = callbacks.ModelCheckpoint('weights/0817_rnn_bin_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = regressor2.fit(X_train2, y_train,
                       validation_data=(X_test2, y_test),
                       epochs=100, batch_size=64, 
                       callbacks=[es, mc]
         )

his = pd.DataFrame.from_dict(history.history)

hist_csv_file = 'weights/0816history.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)