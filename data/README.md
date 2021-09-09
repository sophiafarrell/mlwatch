# Data
Sophia Farrell 
<br>
Date of Creation: 09/08/2021

### Description
This directory contains the data (at least, paths to it), and tools to process this data, used for this project. 

If you are looking for the actual data files themselves, you can find them at: 
```
/p/gpfs1/sfarrell/ml_project/data
```

### Processing root files to FRED
Root files are first simulated (via Marc Bergevin), then processed using a slightly modified version of FRED. 
- Root files in: `/p/gpfs1/adg/path_a/fast_neutron_work/root_files_16m_20pct_rPMT_5700mm_mockData_detectorMedia_doped_water/` for Gd-H20 and `/p/gpfs1/adg/path_a/fast_neutron_work_wbls/root_files_16m_20pct_rPMT_5700mm_mockData_detectorMedia_doped_water/` (and appropriate directories therein)
- Use the load-up executable to reproduce.
  - Using the following FRED versions /p/gpfs1/adg/wmutils/fred_pmt_patterns/ or /p/gpfs1/adg/wmutils/fred_pmt_patterns_wbls/ to reprocess all files 
  - Example: 
  ```
  for f in /p/gpfs1/adg/path_a/fast_neutron_work/root_files_16m_20pct_rPMT_5700mm_mockData_detectorMedia_doped_water/IBDNeutron_LIQUID_ibd_n/*.root; do fred_pmt ${f} /p/gpfs1/sfarrell/data/ibd_neutrons/${f##*/} ; done
  ```
  - Can do `hadd out.root run*.root` to combine multiple run files 
  - This was done for all data in `/p/gpfs1/sfarrell/ml_project/data/root_files`
  
### FRED root files to JSON-type 
Root files aren't intrinsic to python, so aren't super efficient to use, though packages exist to handle this. 
We will use awkward and uproot to handle our root files, converting them to jsons for some purposes
(though you could always skip this and just use awkward to load every time, then make cuts.)
Cuts are made in the process to rid the jsons of the first couple/last couple events in a run, as well as those that had bad fits (e.g., n9 never calculated)


## Function tools to convert files: 
While you can copy my data, it might be more beneficial to use the same tools for other datasets. 
You'll find all these functions in the file: `data_preprocessing.py`. 

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
