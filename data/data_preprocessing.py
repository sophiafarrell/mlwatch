'''
# data_preprocessing
Contains the tools to load up, save, and convert data. 
Contains preprocessing tools. 

### Functions: 

**Data loading/saving/converting**
- root_to_json
- load_json_to_awkward
- get_paired_data
- create_train_test_sets
- load_pickled_data

**PMT position data**
- get_pmt_positions
- load_pmt_positions

**Preprocessing of fred features**
- get_fred_dims
- add_netoutput_to_rf
- scale_features
'''

import uproot 
import awkward as ak
import numpy as np 
import pickle 
from sklearn.model_selection import train_test_split
import pandas as pd

def root_to_json(filename, root_dir = './data/root_files/',
                 save=True, savename=None, save_dir = './data/json_files/',
                 return_data=True
                ):
    '''
    The first step in going from root FRED files to python ML inputs.
    Takes a root file and converts to an awkward array
    Arguments can support saving as a json as well.
    '''
    
    rootfile = uproot.open(f'{root_dir}{filename}.root:data')
    data= rootfile.arrays(library='awkward')

    #timesort the PMT info just to make things easier later
    args = ak.argsort(data['hittime'])
    for key in ['hittime', 'pmtcharge', 'channel']:
        data[key] = data[key][args]

    # Get rid of the first and last events of each run 
    data = data[data['inner_hit_prev']>0]
    data = data[data['inner_hit_next']>0]
    data = data[data['dt_prev_us']>0]

    if save:
        if savename == None: savename = filename
        ak.to_json(data, f'{save_dir}{savename}.json')
        
    if return_data: 
        return data

def load_json_to_awkward(filename, json_dir='./data/json_files/', 
                         
                   ):
    '''
    Take json file and load into awkward array 
    '''
    data = ak.from_json(f'{json_dir}{filename}.json')
    return data 

def get_paired_data(data):
    '''
    For fast-neutron, IBD classification 
    Return data in coupled format, with labels, that pass cuts
    '''
    
    next_ts = data.timestamp[1:]
    next_ts = ak.concatenate([next_ts, 0])

    condition_1 = next_ts - data.timestamp<600
    condition_2 = next_ts - data.timestamp>0
    condition_3 = data.dt_next_us < 600

    both = condition_1 * condition_2 * condition_3

    first_of_two = ak.where(both)[0]
    signal1 = data[first_of_two]
    signal2 = data[first_of_two+1]
    
    return signal1, signal2

def create_train_test_sets(data_list, paired_signals=True,
                           random_state=43, test_size=0.25, 
                           return_data=True, 
                           save=True, savepath='./data/train_test_sets/', savefile=None,
                          ):
    '''
    Create train/test labeled sets from data
    
    data_list: should look like [class0, class1] where class0, class1 are EITHER 
    each an ak array, OR, are each a list of ak arrays (paired signals)
    examples ... data_list = [ibdp, ibdn] OR data_list = [[fn0, fn1], [ibd0, ibd1]]
    
    
    '''
    if paired_signals: 
        selection = [data_list[0][0], data_list[1][0]]
    else: selection = data_list 
    
    #equal split classes (might want to change later)
    samples = min([len(i) for i in selection])
    y = np.append(np.zeros(samples), np.ones(samples))
    trainidx, testidx = train_test_split(np.arange(len(y)), test_size=.25, random_state=43)
    
    if paired_signals:
        first_signal = ak.concatenate([data_list[0][0][:samples], data_list[1][0][:samples]])
        second_signal = ak.concatenate([data_list[0][1][:samples], data_list[1][1][:samples]])
        data_train = dict(ev1 = first_signal[trainidx],
                     ev2 = second_signal[trainidx],
                     y = y[trainidx]
                    )
        data_test = dict(ev1 = first_signal[testidx],
                     ev2 = second_signal[testidx],
                     y = y[testidx]
                    )
    
    else:    
        alldata = ak.concatenate([data_list[0][:samples], data_list[1][:samples]])
        xtrain, xtest = alldata[trainidx], alldata[testidx]
        data_train = dict(x=xtrain, y=y[trainidx])
        data_test = dict(x=xtest, y= y[testidx])
    
    if save:
        pickle.dump(data_train, open( f"{savepath}{savefile}.pkl", "wb" ))
        pickle.dump(data_test, open( f"{savepath}{savefile}.pkl", "wb" ))
    if return_data:
        return data_train, data_test

def load_pickled_data(loadfile, loadpath='./data/train_test_sets/'
                        ):
    '''
    Load pkl file of train or test data 
    '''
    
    data = pickle.load(open(f"{loadpath}{loadfile}.pkl", 'rb'))    
    return data

def get_pmt_positions(rootfiles=['ibd_water', 'ibd_wbls'], root_dir='./data/root_files/',
                      save=False, save_dir='./data/', savefile='pmtpositions',
            ):
    '''
    Get and return or save a dict of PMT xyz positions for both h20 and wbls media 
    ** warning note: n_pmts hard-coded to (h20=2330, wbls=1232)
    '''
    
    n_pmts = dict(h20=2330, wbls=1232)
    out_pos_dict = dict()
    for rootfile, medium, num_pmts in zip(rootfiles, n_pmts.keys(), n_pmts.values()):
        print(f'Processing {rootfile}: Medium: {medium}  Num Pmts: {num_pmts}')
        runinfo = uproot.open(f'{root_dir}{rootfile}.root'+':runSummary')['xyz']       
        pmtpositions = runinfo.arrays(library='np')['xyz']
        pmtpositions = pmtpositions[0][:3*num_pmts]
        pmtxyz = pmtpositions.reshape((num_pmts,3))
        out_pos_dict[medium] = pmtxyz
    if save: pickle.dump(out_pos_dict, open( f'{save_dir}{savefile}.pkl', "wb" ))
    return out_pos_dict

def load_pmt_positions(loadfile='pmtpositions', load_dir='./data/', medium='h20', 
                       
                      ):
    '''
    Load dictionary of pmt positions 
    Args:
    medium: (str) 'h20' or 'wbls'
    '''
    pmtxyz = pickle.load(open( f'{load_dir}{loadfile}.pkl', 'rb'))
    return pmtxyz[medium.lower()]

def get_fred_dims(data, dimensions):
    '''
    Feed a list of dimensions (str) and a dataset to retrieve fred dimensions 
    
    data: should be an awkward array, pd.dataframe, or dictionary. 
    '''
    df_cut = data[dimensions]
    print('Remaining variables selected for analysis: %i'%(len(dimensions)))
    return df_cut

def add_netoutput_to_rf(X_train, X_test, 
              y_train, y_test, fred_dimensions,
              nn_out_train=None, nn_out_test=None,  newdims = ['gcn_out'], 
             ):
    '''
    Add neural net output to RF dimensions
    Good for quick comparison, to see if RF/NN benefit from eachother
    nn_out: either None, or  an array of predictions (n_samples by 1)
    '''
    dims_add = fred_dimensions.copy()
    
    if nn_out_train == None: 
        return X_train, X_test, y_train, y_test, dims_add
    
    new_train = np.hstack((X_train, nn_out_train))
    new_test = np.hstack((X_test, nn_out_test))
    
    for i in newdims:
        dims_add.append(i)    
    return new_train, new_test, y_train, y_test, dims_add

def scale_features(train, test):
    '''
    returns scaler and scaled featuers/y values for FRED variables
    
    train, test: should be of type data generator
    
    return: X1, y_train, X1test, y_test, sc(StandardScaler)
    '''
    X1 = get_fred_dims(train.x, dimensions=dimensions)
    X1 = ak.to_pandas(X1)

    X1test = get_fred_dims(test.x, dimensions=dimensions)
    X1test = ak.to_pandas(X1test)

    sc = StandardScaler()
    X_train3, y_train = sc.fit_transform(X1), train.y
    X_test3, y_test = sc.transform(X1test), test.y
    
    return X1, y_train, X1test, y_test, sc