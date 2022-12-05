import uproot
import numpy as np
import pandas as pd
import awkward as ak
from scipy.stats import norm
import scipy.interpolate as interpolate
from scipy.ndimage import median as med

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Input, InputLayer
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks, regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

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
pmtpos = np.loadtxt('pmtpositions.csv', delimiter=',')
pmtpos = np.vstack((np.array([0, 0, 0]), pmtpos))

fastn = ak.from_json('data/fastn.json')
ibd = ak.from_json('data/ibd.json')

for j, data in enumerate([fastn, ibd]):
    print('set %i'%(j))
    args = ak.argsort(data['restime'])
    for key in ['hittime', 'pmtcharge', 'channel', 'restime']:
        data[key] = data[key][args]

dimensions = [
    'n9', 'n9_prev', 'n9_next',
    'x', 'y', 'z',  'r',
    'id_plus_dr_hit', 'inner_hit_prev', 'inner_hit_next',
    'good_dir', 'good_dir_prev', 'good_dir_next',
    'good_pos','good_pos_prev', 'good_pos_next',
    'closestPMT', 'closestPMT_prev', 'closestPMT_next', 
    'drPrevr', 'dzPrevz', 'drNextr', 'dzNextz',
     'dt_prev_us', 'dt_next_us',
    'azimuth_ks', 'azimuth_ks_prev','azimuth_ks_next',
    'n100', 'pe', 
    'beta_one', 'beta_two', 'beta_three', 'beta_four', 'beta_five', 'beta_six',
    'beta_one_prev', 'beta_two_prev', 'beta_three_prev', 'beta_four_prev', 'beta_five_prev', 'beta_six_prev',
]
def get_dims(data, dimensions=dimensions):
    df_cut = data[dimensions]
    print('Remaining variables selected for analysis: %i'%(len(dimensions)))
    return df_cut

delaytime = 1e5
c = 21.8 #cm/ns

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
        
        xyz = pmtpos[chn]+1 #since value of 0 has meaning..
        
        time = sequence.pad_sequences(dat[keys[2]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
        new[:,:,0] = chrg
        new[:,:,1] = time    
        new[:,:,2:5] = xyz+1
        
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
    return X, y

all_data = ak.concatenate([ibd, fastn])

fn = all_data['code']==2
invbeta = all_data['code']==1
id0 = all_data['subid']==0
id1p = all_data['subid']>0 # >0 changes things 

ibd0_mask = ak.from_iter([all(t) for t in zip(invbeta, id0)])
ibd1_mask = ak.from_iter([all(t) for t in zip(invbeta, id1p)])

samples = min([ak.count_nonzero(fn), 
              ak.count_nonzero(ibd0_mask), 
              ak.count_nonzero(ibd1_mask)])

x1 = all_data[fn][:samples]
x1 = ak.concatenate((x1, all_data[ibd0_mask][:samples]))
x1 = ak.concatenate((x1, all_data[ibd1_mask][:samples]))

data_to_manipulate = [all_data[fn][:samples],
                     all_data[ibd0_mask][:samples], all_data[ibd1_mask][:samples]
                     ]

X1 = get_dims(x1, dimensions=dimensions)
X1 = ak.to_pandas(X1)
X2, y = prep_net_data(data_to_manipulate, prompt=False, restime=True, maxlen=200)
a = np.array(y.astype(int))
b = np.zeros((a.size, a.max()+1), dtype='int32')
b[np.arange(a.size),a] = 1
y = b

# Split dataset into training set and test setd
X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(X1, X2, y, test_size=0.25, random_state=43) 
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

def model_builder(hp):
    # units for first dense layer 
#     units_1 = hp.Int('units', min_value=256, max_value=512, step=10)
#     # Units for LSTM layers 
#     units_2 = hp.Int('units', min_value=128, max_value=512, step=10)
    # Learning rate 
    hp_learning_rate = hp.Choice('learning_rate', values=[3e-4, 1e-4])
    hp_regularizer = hp.Choice('activity_regularizer', values=[1e-2, 1e-3])

    model1 = Sequential()
    # create the model
    model1.add(InputLayer(42))
    model1.add(Dense(units=256, 
                     activation='relu', 
                     activity_regularizer=regularizers.l2(hp_regularizer)))
    model1.add(Dropout(0.3))

    regressor2 = Sequential()
    input_shape = X2.shape[1:]
    regressor2.add(InputLayer(input_shape=input_shape))
    regressor2.add(LSTM(units=512, 
                       return_sequences=True, 
                       dropout=0.4,
                      ))  
    regressor2.add(LSTM(units=256, 
                       return_sequences=True, 
                       dropout=0.4,
                      )) 
    regressor2.add(LSTM(units=128, 
                       return_sequences=True, 
                       dropout=0.4, 

                      ))
    regressor2.add(LSTM(units=128, 
                       return_sequences=False, 
                       dropout=0.4, 

                      ))

    mergedOut = Concatenate()([model1.output, regressor2.output])
    mergedOut = Flatten()(mergedOut)    
    mergedOut = Dense(256, activation='relu', activity_regularizer=regularizers.l2(hp_regularizer))(mergedOut)
    mergedOut = Dropout(.3)(mergedOut)
    mergedOut = Dense(128, activation='relu', activity_regularizer=regularizers.l2(hp_regularizer))(mergedOut)
    mergedOut = Dropout(.3)(mergedOut)
    mergedOut = Dense(64, activation='relu')(mergedOut)
    mergedOut = Dropout(.3)(mergedOut)
    mergedOut = Dense(3, activation='softmax')(mergedOut)

    newModel = Model([model1.input, regressor2.input], mergedOut)
    newModel.compile(loss='categorical_crossentropy', 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                     metrics=['accuracy']
                    )
    return newModel

# Instantiate the tuner
tuner = kt.Hyperband(model_builder, # the hypermodel
                     objective='val_accuracy', # objective to optimize
max_epochs=20,
factor=3, # factor which you have seen above 
directory='logs', # directory to save logs 
project_name='resorted_res_deep')
print('Tensorflow Version: %s'%(tf.__version__))
tuner.search_space_summary() 

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)
mc = callbacks.ModelCheckpoint('weights/0812scan_best_model_3class.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = tuner.search([X_train1, X_train2], y_train,
                       validation_data=([X_test1, X_test2], y_test),
                       epochs=100, batch_size=64, 
                       callbacks=[es, mc]
         )

# hist_csv_file = 'weights/history.csv'
# with open(hist_csv_file, mode='w') as f:
#     history.to_csv(f)