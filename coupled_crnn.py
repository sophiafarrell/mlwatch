exec(open("./do_imports.py").read())

selection = []
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

npmts = 2330
timetot = 1500 #ns 
tres = 10 # ns 
div_factor = tres 
nbins = int(timetot/tres)
y = np.array([]) 

fn0, fn1 = get_paired_data(fastn)
ibd0, ibd1 = get_paired_data(ibd)

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

del first, second, data_to_manipulate, new_bins1, new_bins2, fastn, ibd, fn0, fn1, ibd0, ibd1
gc.collect()

## MODEL BUILDING:
# go to models/lstm_to_merged_crnn.py
# exec(open("./lstm_to_merged_crnn.py").read())

nunits = 128
## MODEL SIDE 1 
lstm1 = Sequential()
lstm1.add(InputLayer(input_shape=X_train1.shape[1:]))
forward_layer = LSTM(nunits, return_sequences=True, dropout=0.05)
backward_layer = LSTM(nunits, return_sequences=True, go_backwards=True, dropout=0.05)
lstm1.add(Bidirectional(forward_layer, backward_layer=backward_layer))
lstm1.add(Reshape((2330,2*nunits,1),input_shape=(2330,2*nunits)))
# lstm1.add(Conv2D(32, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
#                 )
#          )
lstm1.add(MaxPooling2D(pool_size=(2, 2)))
## MODEL SIDE 2
lstm2 = Sequential()
lstm2.add(InputLayer(input_shape=X_train2.shape[1:]))
forward_layer2 = LSTM(nunits, return_sequences=True, dropout=0.05)
backward_layer2 = LSTM(nunits, return_sequences=True, go_backwards=True, dropout=0.05)
lstm2.add(Bidirectional(forward_layer2, backward_layer=backward_layer2))
lstm2.add(Reshape((2330,2*nunits,1),input_shape=(2330,2*nunits)))
# lstm2.add(Conv2D(32, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
#                 )
#          )
lstm2.add(MaxPooling2D(pool_size=(2, 2)))

## MERGED MODEL
merged = Concatenate(axis=2)([lstm1.output, lstm2.output])
merged = Conv2D(32, (3,3), 
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Conv2D(64, (3,3), 
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)
merged = Conv2D(64, (3,3), 
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)
merged = Conv2D(64, (3,3), 
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Conv2D(64, (3,3), 
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Flatten()(merged)  
merged = Dense(64, activation='relu',  
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
              )(merged)
merged = Dense(32, activation='relu',  
               kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
              )(merged)

merged = Dense(1, activation='sigmoid')(merged)

## COMPILE AND TRAIN TOTAL MODEL 
newModel = Model([lstm1.input, lstm2.input], merged)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=350,
    decay_rate=0.95)

newModel.compile(loss='binary_crossentropy', 
                 optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                 metrics=['accuracy']
                )
newModel.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = callbacks.ModelCheckpoint('weights/coupled_crnn_4.h5', monitor='val_accuracy', mode='max', save_best_only=True)

history = newModel.fit([X_train1, X_train2], y_train,
                       validation_data=([X_test1,X_test2], y_test),
                       epochs=100, batch_size=24, 
                       callbacks=[es, mc]
         )

his = pd.DataFrame.from_dict(history.history)
hist_csv_file = 'weights/history_coupled_crnn.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)