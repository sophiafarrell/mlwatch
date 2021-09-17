exec(open("./do_imports.py").read())

import cnn_batched_data
import argparse 

## MODEL BUILDING:
# go to models/lstm_to_merged_crnn.py
# exec(open("./lstm_to_merged_crnn.py").read())
def get_model(in_shape=(2330, 150)):
    nunits = 128
    nunits2 = 64
    ## MODEL SIDE 1 
    lstm1 = Sequential()
    lstm1.add(InputLayer(input_shape=in_shape))
    lstm1.add(Reshape((in_shape[0],in_shape[1],1),input_shape=(in_shape[0],in_shape[1])))
    lstm1.add(Conv2D(128, (10,10), ))
    lstm1.add(Conv2D(64, (3,3), ))
    lstm1.add(Conv2D(64, (3,3), ))
    lstm1.add(MaxPooling2D(pool_size=(2, 2)))
    lstm1.add(Conv2D(64, (3,3), ))
#     lstm1.add(MaxPooling2D(pool_size=(2, 2)))
    
    ## MODEL SIDE 2
    lstm2 = Sequential()
    lstm2.add(InputLayer(input_shape=in_shape))
    lstm2.add(Reshape((in_shape[0],in_shape[1],1),input_shape=(in_shape[0],in_shape[1])))
    lstm2.add(Conv2D(128, (10,10), ))
    lstm2.add(Conv2D(64, (3,3), ))
    lstm2.add(Conv2D(64, (3,3), ))
    lstm2.add(MaxPooling2D(pool_size=(2, 2)))
    lstm2.add(Conv2D(64, (3,3), ))
#     lstm2.add(MaxPooling2D(pool_size=(2, 2)))
    ## MERGED MODEL
    merged = Concatenate(axis=-1)([lstm1.output, lstm2.output])
#     merged = Conv2D(64, (3,3), )(merged)
# #     merged = MaxPooling2D(pool_size=(2, 1))(merged)
#     merged = Conv2D(64, (3,3),)(merged)
# #     merged = MaxPooling2D(pool_size=(2, 1))(merged)
#     merged = Conv2D(64, (3,3),)(merged)
#     merged = MaxPooling2D(pool_size=(2, 2))(merged)

    merged = Flatten()(merged)  
    merged = Dense(64, activation='relu',)(merged)
    merged = Dense(32, activation='relu',)(merged)

    merged = Dense(1, activation='sigmoid')(merged)

    ## COMPILE AND TRAIN TOTAL MODEL 
    FullModel = Model([lstm1.input, lstm2.input], merged)
    
    return FullModel


def NNTraining(batch_size, epochs):
    train_generator = cnn_batched_data.DatSequence('train_water', batch_size=batch_size)
    test_generator = cnn_batched_data.DatSequence('test_water', batch_size=batch_size)

    in_shape=train_generator.out_shape
    model = get_model(in_shape=in_shape) 

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=200,
        decay_rate=0.95)
    adam = tf.keras.optimizers.Adam(learning_rate=.001)
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = callbacks.ModelCheckpoint('weights/coupled_cnn.h5', 
                                   monitor='val_accuracy', mode='max', save_best_only=True)
    
    history = model.fit(train_generator,
                       steps_per_epoch = int(train_generator.n_samples // batch_size),
                       validation_data=test_generator,
                       validation_steps = int(test_generator.n_samples // batch_size),
                       epochs=epochs, 
                       workers=10,
                       use_multiprocessing=True,
                       callbacks=[es, mc],
         )

    his = pd.DataFrame.from_dict(history.history)
    hist_csv_file = 'weights/history_coupled_crnn.csv'
    with open(hist_csv_file, mode='w') as f:
        his.to_csv(f)
    return 0 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, nargs='?', default=64)
    parser.add_argument('--epochs', dest='epochs', type=int, nargs='?', default=20)
    args = parser.parse_args()
#     tf.config.threading.set_inter_op_parallelism_threads(50)
#     tf.config.threading.set_intra_op_parallelism_threads(50)
    print('Hello human... you want to do some ML')
    NNTraining(args.batch_size, args.epochs)
    print('bye')
