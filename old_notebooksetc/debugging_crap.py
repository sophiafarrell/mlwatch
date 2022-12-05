exec(open("./do_imports.py").read())

import batched_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

batch_size=32
train_generator = batched_data.DatSequence('train_water', batch_size=batch_size)
test_generator = batched_data.DatSequence('test_water', batch_size=batch_size)


## MODEL BUILDING:
# go to models/lstm_to_merged_crnn.py
# exec(open("./lstm_to_merged_crnn.py").read())

nunits = 128
## MODEL SIDE 1 
lstm1 = Sequential()
lstm1.add(InputLayer(input_shape=(train_generator.npmts, train_generator.nbins)))
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
lstm2.add(InputLayer(input_shape=(train_generator.npmts, train_generator.nbins)))
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
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Conv2D(64, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)
merged = Conv2D(64, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)
merged = Conv2D(64, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Conv2D(32, (3,3), 
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
               )(merged)
merged = MaxPooling2D(pool_size=(2, 2))(merged)

merged = Flatten()(merged)  
merged = Dense(64, activation='relu',  
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
              )(merged)
merged = Dense(32, activation='relu',  
#                kernel_regularizer=tf.keras.regularizers.l2(1e-5), 
              )(merged)

merged = Dense(1, activation='sigmoid')(merged)

## COMPILE AND TRAIN TOTAL MODEL 
newModel = Model([lstm1.input, lstm2.input], merged)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=200,
    decay_rate=0.95)

newModel.compile(loss='binary_crossentropy', 
                 optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                 metrics=['accuracy']
                )
newModel.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = callbacks.ModelCheckpoint('weights/coupled_crnn_more.h5', monitor='val_accuracy', mode='max', save_best_only=True)
print(int(test_generator.n_samples // batch_size))
history = newModel.fit(train_generator,
                       steps_per_epoch = int(train_generator.n_samples // batch_size),
                       validation_data=test_generator,
                       validation_steps = int(test_generator.n_samples // batch_size),
                       epochs=100, 
                       workers=10,
                       use_multiprocessing=True,
                       callbacks=[es, mc],
         )

his = pd.DataFrame.from_dict(history.history)
hist_csv_file = 'weights/history_coupled_crnn.csv'
with open(hist_csv_file, mode='w') as f:
    his.to_csv(f)
    