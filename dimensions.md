# Dimensions
Sophia Farrell; July 2021

### Key: 
- **bold**: Something unknown still
- ??: Unknown clarification
- --: N/A or no units 

| Variable | Description | Units |
| --- | --- | --- |
| gtid | Event ID in frame | -- |
| mcid | Eveny ID in MC process | -- |
| subid | Sub-event id (0 is the first) | -- |
| inner_hit | Number of PMT hits from _inner detector_ | **Hits**: #triggered PMTs |
| inner_hit_prev | inner_hit from previous event <br> (same of all X_prev variables) | hits |
| id_plus_dr_hit | inner detector plus dark rate hits | hits |  
| veto_hit | # hits from veto PMTs | hits |
| veto_plus_dr_hit | # hits from veto PMTs + dark rate | hits | 
| pe | total area from all hits. formally, integrated charge of hits | PE (photoelectrons) | 
| innerPE | area from inner_hit | PE (photoelectrons) | 
| vetoPE | area from veto_hit | PE (photoelectrons) | 
| n9 | number of PMTs where prompt light was collected <br> have within a (+6ns and -3ns) residual time <br> from t0 (reconstructed vertex time) | residuals (like hits) |
| **nOff** | ?? | hits |
| n100 | 100ns window of residuals | residuals (hits) |
| n400 | 400ns window of residual PMTs | residuals (hits) |
| nX | variable prompt residual measurement | residuals (hits) |
| **good_dir** | some GOF on the direction of the cone? | ?? |
| **good_pos** | some GOF on the vertex position? | ?? |
| x, y, z | vertex (reconstructed) | mm |
| t | reconstructed vertex time | **ns?** |
| u, v, w | Cherenkov cone components (not well understood yet) | ?? |
| **azimuth_ks** | ks test on azimuthal measurement (isotropy) | -- |
| **distpmt** | proximity to PMT wall parallel to z-position of vertex | mm |
| closestPMT | proximity to PMT wall | mm |
| mc_energy | True Energy of infcoming particle (e.g. fast neutron) | MeV | 
| mcx, mcy, mcz | Truth position | mm |
| mct, mcu, mcv, mcw | Truth t, u, v, w variables | ns, -- |
| dxPrevx, dyPrevy, dzPrevz, drPrevr | difference in reco position between this and prev (sub-)event | mm |
| drPrevrQFit | diff in r position using QFit algo | mm |
| dxmcx, dxmcy, dxmcz, drmcr | diff in MC truth/reconstructed position | mm |
| dt_sub | time of the sub-event trigger from start of the event mc | us |
| dt_prev_us | time difference between last event | us | 
| timestamp | Time of event since run start | us? |
| num_tested | # tested points in the likelihood calc from BONSAI | -- |
| best_like | the best log-likelihood of the points | -- |
| worst_like | the worst log-likelihood of the points | -- |
| average_like | the average log-likelihood of the points | -- |
| average_like_05m | the average log likelihood excluding a 0.5m sphere around the best fit | -- |