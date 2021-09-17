exec(open("./do_imports.py").read())

def prep_net_data_simple(data_to_manipulate, 
                  keys = ['channel', 'pmtcharge', 'hittime'],
                  maxlen=200):
    X = []
    y = np.array([])        
    diminput = len(keys)
    for d, dat in enumerate(data_to_manipulate):
        print('Running set %i of %i'%(d+1, len(data_to_manipulate)))
        new = np.empty(shape=(len(dat), maxlen, diminput))

        new[:,:,0] = sequence.pad_sequences(dat[keys[0]]+1, maxlen=maxlen, 
                                     padding='pre', 
                                     dtype='int32')
        new[:,:,1] = sequence.pad_sequences(dat[keys[1]], maxlen=maxlen, 
                                      padding='pre', 
                                      dtype='float64')
        
        
        new[:,:,2] = sequence.pad_sequences(dat[keys[2]], maxlen=maxlen, 
                                      padding='pre', 
                                      dtype='float64')            
        if len(X)>0:
            X = np.append(X, new, axis=0)
        else: 
            X = new
        y=np.append(y, d*np.ones(len(new)))   
    if np.max(y)>1:
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
    return X, y


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
X2, y = prep_net_data(data_to_manipulate, keys = ['channel', 'pmtcharge', 'restime'], maxlen=200)
# X2, y = prep_net_data_simple(data_to_manipulate)
# Split dataset into training set and test set
X_train2, X_test2, y_train, y_test, = train_test_split(X2, y,
                                                     test_size=0.25, random_state=43) 



## MODEL BUILDING 
regressor2 = Sequential()
input_shape = X2.shape[1:]
regressor2.add(InputLayer(input_shape=input_shape))
regressor2.add(LSTM(units=256, 
                    return_sequences=True, 
                    dropout=0.2,
                  ))
regressor2.add(LSTM(units=128, 
                   return_sequences=True, 
                   dropout=0.2,
                  ))
regressor2.add(LSTM(units=64, 
                   return_sequences=True, 
                   dropout=0.2, 
                  ))
regressor2.add(LSTM(units=32, 
                    return_sequences=True, 
                    dropout=0.2, 
                   ))
regressor2.add(LSTM(units=16, 
                    return_sequences=True, 
                    dropout=0.2, 
                   ))

regressor2.add(Flatten())

regressor2.add(Dense(256))
regressor2.add(LeakyReLU(alpha=0.05))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(128))
regressor2.add(LeakyReLU(alpha=0.05))
regressor2.add(Dropout(0.2))

# regressor2.add(Dense(64))
# regressor2.add(LeakyReLU(alpha=0.05))
# regressor2.add(Dropout(0.2))

regressor2.add(Dense(32))
regressor2.add(LeakyReLU(alpha=0.05))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(1, activation='sigmoid'))

regressor2.compile(loss='binary_crossentropy', 
                 optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4),
                 metrics=['accuracy']
                )
regressor2.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
mc = callbacks.ModelCheckpoint('weights/cherenkov_model_1.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = regressor2.fit(X_train2, y_train,
                       validation_data=(X_test2, y_test),
                       epochs=100, batch_size=64, 
                       callbacks=[es, mc]
         )

his = pd.DataFrame.from_dict(history.history)

hist_csv_file = 'weights/cherenkov_history.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)