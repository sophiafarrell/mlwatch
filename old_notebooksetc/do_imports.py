# exec(open("./setup_fd_er.py").read())
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import uproot
import pickle

import numpy as np
import pandas as pd
import awkward as ak
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, InputLayer, Bidirectional
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Reshape, AveragePooling2D, Concatenate, Flatten, LeakyReLU
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
        gpu = gpus[1]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        print(e)

def get_dims(data, dimensions):
    df_cut = data[dimensions]
    print('Remaining variables selected for analysis: %i'%(len(dimensions)))
    return df_cut
