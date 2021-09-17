exec(open("./do_imports.py").read())
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape
selection = []

for d, data in enumerate([fastn, ibd]):
    subid0 = data.subid==0
    subid1 = data.subid>=1
    quicknext = data.dt_next_us < 500
    quickprev = data.dt_prev_us < 500
    longprev = data.dt_prev_us > 500
    
    if d==0:
        mask0 = ak.from_iter([all(t) for t in zip(quicknext, longprev)])
        mask1 = ak.from_iter([all(t) for t in zip(quickprev, subid1)])
    else: 
        mask0 = ak.from_iter([all(t) for t in zip(quicknext, subid0)])
        mask1 = ak.from_iter([all(t) for t in zip(quickprev, subid1)])
    signal0 = data[mask0]
    signal1 = data[mask1]
    
    selection.append(signal0)
    selection.append(signal1)
    
samples = min([len(i) for i in selection])
smth = [i[:samples] for i in selection]
data_to_manipulate = [ak.concatenate(smth[:2]), ak.concatenate(smth[2:])]
samples = min([len(i) for i in data_to_manipulate])

# nbins = 100
# div_factor = 1500/nbins
npmts = 2330
timetot = 1500 #ns 
tres = 10 # ns 
div_factor = tres 
nbins = int(timetot/tres)
y = np.array([]) 

newdata = np.zeros((2*samples, npmts, nbins), dtype=np.float32)
print(newdata.shape)
for d, data in enumerate(data_to_manipulate):
    print('set %i of %i'%(d+1, len(data_to_manipulate)))
    new_bins = ak.values_astype((data.hittime-300)/div_factor, np.int32)
    y=np.append(y, d*np.ones(len(new_bins)))   
    for i in range(0, samples):
        newdata[(d+1)*i, data.channel[i], new_bins[i]]=data.pmtcharge[i]
        
X_train2, X_test2, y_train, y_test, = train_test_split(newdata, y,
                                                     test_size=0.25, random_state=43) 
del newdata
## MODEL BUILDING: 
regressor2 = Sequential()
regressor2.add(InputLayer(input_shape=X_train2.shape[1:]))

# regressor2.add(LSTM(units=256, 
#                     return_sequences=True, 
#                     dropout=0.2,
#                   ))
regressor2.add(LSTM(units=128, 
                    return_sequences=True, 
                    dropout=0.2,
                  ))
regressor2.add(LSTM(units=64, 
                    return_sequences=True, 
                    dropout=0.2,
                  ))

regressor2.add(Reshape((466,320,1),input_shape=(2330,64)))
regressor2.add(Conv2D(64, (3,3)))
regressor2.add(Dropout(0.2))
regressor2.add(MaxPooling2D(pool_size=(2, 2)))

regressor2.add(Conv2D(64, (3,3)))
regressor2.add(Dropout(0.1))
regressor2.add(MaxPooling2D(pool_size=(2, 2)))

regressor2.add(Conv2D(64, (3,3)))
regressor2.add(Dropout(0.2))
regressor2.add(MaxPooling2D(pool_size=(2, 2)))

regressor2.add(Conv2D(32, (3,3)))
regressor2.add(Dropout(0.2))
regressor2.add(MaxPooling2D(pool_size=(2, 2)))

regressor2.add(Flatten())
regressor2.add(Dense(64, activation='relu'))
regressor2.add(Dropout(0.2))
regressor2.add(Dense(32, activation='relu'))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(1, activation='sigmoid'))

regressor2.compile(loss='binary_crossentropy', 
                 optimizer= 'adam', # tf.keras.optimizers.Adam(learning_rate=1e-4),
                 metrics=['accuracy']
                )
regressor2.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
mc = callbacks.ModelCheckpoint('weights/fn_ibd_cnnmodel_semi.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = regressor2.fit(X_train2, y_train,
                       validation_data=(X_test2, y_test),
                       epochs=100, batch_size=32, 
                       callbacks=[es, mc]
         )

his = pd.DataFrame.from_dict(history.history)
hist_csv_file = 'weights/cnn_history_semi.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)