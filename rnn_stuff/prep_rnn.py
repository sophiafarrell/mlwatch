exec(open("./do_imports.py").read())

def get_paired_data(data):
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
ibd = ak.from_json('data/ibd.json')
fastn = ak.from_json(fastnwbls, 'data/fastn_water.json')

fn0, fn1 = get_paired_data(fastn)
ibd0, ibd1 = get_paired_data(ibd)

npmts = 2330
timetot = 1500 #ns 
tres = 10 # ns 
div_factor = tres 
nbins = int(timetot/tres)
y = np.array([]) 


selection = [ibd0, fn0]
samples = min([len(i) for i in selection])
size = samples * len(selection)

first = np.zeros((size, npmts, nbins), dtype=np.float32)
second = np.zeros_like(first)

data_to_manipulate = [[fn0, fn1], [ibd0, ibd1]]

for d, (signal1, signal2) in enumerate(data_to_manipulate):
    print('set %i of %i'%(d+1, len(data_to_manipulate)))
    
    new_bins1 = ak.values_astype((signal1.hittime-300)/div_factor, np.int32)
    new_bins2 = ak.values_astype((signal2.hittime-300)/div_factor, np.int32)

    for i in range(0, samples):
        first[(d+1)*i, signal1.channel[i], new_bins1[i]]=signal1.pmtcharge[i]
        second[(d+1)*i, signal2.channel[i], new_bins2[i]]=signal2.pmtcharge[i]
    y=np.append(y, d*np.ones(samples))
    

X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(
    first, second, y,
    test_size=0.25, random_state=43
)

data_train = dict(X1=X_train1, X2=X_train2,
            y=y_train
           )
data_test = dict(X1=X_test1, X2=X_test2,
            y=y_test
           )
# np.save('water_train.npy', data_train)
# np.save('water_test.npy', data_test)

pickle.dump( data_test, open( "water_test.pkl", "wb" ), protocol=4 )
pickle.dump( data_train, open( "water_train.pkl", "wb" ), protocol=4 )