#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import pandas as pd
import numpy as np

import lib.Core.ROOTProcessor as rp

DATA_DIR = "./Data/"
OUTFILE_NAME = "b_fastneutrons"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

if __name__=='__main__':
    data_list = glob.glob(DATA_DIR+"*.root")
    Processor = rp.ROOTProcessor(treename="data")
    for f1 in data_list:
        Processor.addROOTFile(f1)
    data_dictionary = Processor.getProcessedData()
    df = pd.DataFrame(data_dictionary)
    print(df.head())

    #Pickle all dataframes
    df.to_pickle("./%s.pkl"%(OUTFILE_NAME))

