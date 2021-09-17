exec(open("./do_imports.py").read())
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape

quicknext = ibd.dt_next_us < 500
subid0 = ibd.subid==0
subid1 = ibd.subid==1
quickprev = ibd.dt_prev_us < 500

eplus_mask = ak.from_iter([all(t) for t in zip(quicknext, subid0)])
nc_mask = ak.from_iter([all(t) for t in zip(quickprev, subid1)])

eplus = ibd[eplus_mask]
nc = ibd[nc_mask]

# all_data = ak.concatenate([eplus, nc])
samples = min([len(eplus), 
                       len(nc)
                      ])
data_to_manipulate = [eplus[:samples],
                      nc[:samples]
                     ]

npmts = 2330
nbins = 100 
div_factor = 1500/nbins
y = np.array([]) 

newdata = np.zeros((2*samples, npmts, nbins))
print(newdata.shape)
for d, data in enumerate(data_to_manipulate):
    print('set %i of %i'%(d+1, len(data_to_manipulate)))
    new_bins = ak.values_astype((data.hittime-300)/div_factor, np.int32)
    y=np.append(y, d*np.ones(len(new_bins)))   
    for i in range(0, samples):
        newdata[(d+1)*i, data.channel[i], new_bins[i]]=data.restime[i]
        
X_train2, X_test2, y_train, y_test, = train_test_split(newdata, y,
                                                     test_size=0.25, random_state=43) 

## MODEL BUILDING 
regressor2 = Sequential()
regressor2.add(InputLayer(input_shape=X_train2.shape[1:]))
regressor2.add(LSTM(units=128, 
                    return_sequences=True, 
                    dropout=0.2,
                  ))
regressor2.add(Reshape((2330,128,1),input_shape=(2330,128)))
regressor2.add(Conv2D(32, (5,5)))
regressor2.add(MaxPooling2D(pool_size=(4, 2)))

regressor2.add(Conv2D(64, (5,5)))
regressor2.add(MaxPooling2D(pool_size=(4, 2)))

regressor2.add(Conv2D(64, (3,3)))
regressor2.add(MaxPooling2D(pool_size=(4, 2)))

regressor2.add(Conv2D(64, (3,3)))
regressor2.add(MaxPooling2D(pool_size=(2, 2)))

regressor2.add(Flatten())
# regressor2.add(Dense(128))
regressor2.add(Dense(32))

regressor2.add(Dense(1, activation='sigmoid'))

regressor2.compile(loss='binary_crossentropy', 
                 optimizer= 'adam',
                 metrics=['accuracy']
                )
regressor2.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
mc = callbacks.ModelCheckpoint('weights/cnn_model3.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = regressor2.fit(X_train2, y_train,
                       validation_data=(X_test2, y_test),
                       epochs=100, batch_size=32, 
                       callbacks=[es, mc]
         )

his = pd.DataFrame.from_dict(history.history)
hist_csv_file = 'weights/cnn_history.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)