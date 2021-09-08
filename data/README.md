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


### Train and test sets 

### ... 