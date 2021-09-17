exec(open("./do_imports.py").read())

import cnn_batched_data
import argparse 

## MODEL BUILDING:
# go to models/lstm_to_merged_crnn.py
# exec(open("./lstm_to_merged_crnn.py").read())
def get_model(max_length=200, embedds=32):
    lstmlen=32
    model1 = Sequential()
    model1.add(Input(shape=(max_length)))
    model1.add(tf.keras.layers.Embedding(2330, embedds, input_length=max_length, 
#                                        embeddings_regularizer=tf.keras.regularizers.l1(l1=1e-4),
                                       )
             )
    model1.add(LSTM(lstmlen, return_sequences=False, dropout=0.05))
#     model1.add(Reshape((max_length,embedds,1),input_shape=(max_length,embedds)))
# #     model.add(Reshape((max_length,lstmlen,1),input_shape=(max_length,lstmlen)))
#     model1.add(Conv2D(64, (3,3), ))
# #     model.add(Dropout(0.3))
#     model1.add(MaxPooling2D(2,2))
#     model1.add(Conv2D(64, (3,3), ))
# #     model.add(Dropout(0.3))
#     model1.add(MaxPooling2D(2,2))

    
    model2 = Sequential()    
    model2.add(Input(shape=(max_length)))
    model2.add(tf.keras.layers.Embedding(2330, embedds, input_length=max_length, 
#                                        embeddings_regularizer=tf.keras.regularizers.l1(l1=1e-4),
                                       )
             )
    model2.add(LSTM(lstmlen, return_sequences=False, dropout=0.05))
#     model2.add(Reshape((max_length,embedds,1),input_shape=(max_length,embedds)))
#     model.add(Reshape((max_length,lstmlen,1),input_shape=(max_length,lstmlen)))

#     model2.add(Conv2D(64, (3,3), ))
# #     model.add(Dropout(0.3))
#     model2.add(MaxPooling2D(2,2))
#     model2.add(Conv2D(64, (3,3), ))
# #     model.add(Dropout(0.3))
#     model2.add(MaxPooling2D(2,2))

    merged = Concatenate(axis=-1)([model1.output, model2.output])
#     merged = Conv2D(64, (3,3), )(merged)
#     merged = MaxPooling2D(pool_size=(2, 2))(merged)
    
    merged = Flatten()(merged)    
    merged = Dense(32, activation='relu', 
#                    kernel_regularizer='l1',
                   )(merged)
    merged = Dense(1, activation='sigmoid')(merged)
    model = Model([model1.input, model2.input], merged)

    
    return model


def NNTraining(batch_size, epochs):
    train_generator = cnn_batched_data.EmbeddingSeq('train_water', batch_size=batch_size)
    test_generator = cnn_batched_data.EmbeddingSeq('test_water', batch_size=batch_size)

    model = get_model(max_length=train_generator.max_length) 

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=200,
        decay_rate=0.95)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = callbacks.ModelCheckpoint('weights/embedder.h5', 
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
