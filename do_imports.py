# exec(open("./setup_fd_er.py").read())
import gc
import uproot
import numpy as np
import pandas as pd
import awkward as ak
from scipy.stats import norm
import scipy.interpolate as interpolate
from scipy.ndimage import median as med
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Input, InputLayer, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, AveragePooling2D
from tensorflow.keras.layers import Concatenate, Flatten, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks, regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print(tf.__version__, tf.__file__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        print(e)

#Setup and preprocess data 
pmtpos = np.loadtxt('pmtpositions.csv', delimiter=',')+1.

npmt = pmtpos.shape[0]
pmtr = np.sqrt(pmtpos[:,0]**2 + pmtpos[:,1]**2).reshape(npmt, 1)
pmttheta = np.arctan2(pmtpos[:,1], pmtpos[:,0]).reshape(npmt, 1)
pmtz = pmtpos[:,2].reshape(npmt, 1)
pmtrad = np.hstack((pmtr, pmttheta, pmtz))
maxes = np.max(pmtrad, axis=0) 
mins = np.min(pmtrad, axis=0)
normed_pos = (pmtrad-mins)/(maxes-mins)

pmtpos = np.vstack((np.array([0, 0, 0]), pmtpos))
normed_pos = np.vstack((np.array([0,0,0]), normed_pos))

fastn = ak.from_json('data/fastn.json')
ibd = ak.from_json('data/ibd.json')
ibd['norm_hittime'] = (ibd.hittime - 300)/(1800-300)
fastn['norm_hittime'] = (fastn.hittime - 300)/(1800-300)

def get_dims(data, dimensions):
    df_cut = data[dimensions]
    print('Remaining variables selected for analysis: %i'%(len(dimensions)))
    return df_cut

def prep_net_data(data_to_manipulate, 
                  keys = ['channel', 'pmtcharge', 'hittime', 'restime'],
                  maxlen=200):
    X = []
    y = np.array([])        
    diminput = len(keys)
    if 'channel' in keys: diminput+=2
    for d, dat in enumerate(data_to_manipulate):
        print('Running set %i of %i'%(d+1, len(data_to_manipulate)))
        new = np.empty(shape=(len(dat), maxlen, diminput))

        chn = sequence.pad_sequences(dat[keys[0]]+1, maxlen=maxlen, 
                                     padding='post', 
                                     dtype='int32')
        chrg = sequence.pad_sequences(dat[keys[1]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
        
        xyz = pmtpos[chn] #since value of 0 has meaning..
        rthetaz = normed_pos[chn]
        time = sequence.pad_sequences(dat[keys[2]], maxlen=maxlen, 
                                      padding='post', 
                                      dtype='float64')
        new[:,:,0] = chrg
        new[:,:,1] = time    
        new[:,:,2:5] = rthetaz
        
        if 'restime' in keys and len(keys)>3:
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
