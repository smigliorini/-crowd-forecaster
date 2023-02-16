import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


file_name = '../data/export_61.csv'
epochs = 300
nodes = 256
drop = 0.6
train = 'raw' 
#train = 'context'


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(nodes, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(nodes, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(1, activation='linear'),
    ])
    model.summary()
    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    metrics=['mape'])
    return model

def plot_loss(history, title):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('Error [Presenze]')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    raw_dataset = pd.read_csv(file_name)

    if train == 'raw':
        raw_dataset = raw_dataset[['presenze','month','day','doy','week','hour']]

    else:
        raw_dataset = raw_dataset[['presenze','month','day','doy','dow','week','hour','holiday','festive',\
                                    'temperature','humidity','pressure','wind','rain']]


    dataset = raw_dataset.copy()
    dataset.tail()

    dataset.isna().sum() 

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('presenze')
    test_labels = test_features.pop('presenze')


    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    normalizer.mean.numpy()

    dnn_model = build_and_compile_model(normalizer)

    history = dnn_model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        verbose=0,
        validation_split = 0.2)

    #dnn_model.save('model_{}_{}_{}'.format(nodes, drop, epochs))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    print(hist.tail())

    test_results = {}
    test_results = dnn_model.evaluate(
        test_features, test_labels, verbose=1)


        
    img_title = '{} - {}'.format(file_name, test_results)
    plot_loss(history, img_title)


    title = '\n---------------result--------------------\nFile name: {}\nData tipe: {}\n'.format(file_name, train)
    dnn = 'nodes: {} \t dropout: {} \t epochs: {}\n'.format(nodes, drop, epochs)
    result = 'MAPE: {}\n'.format(round(test_results[1],1))

    print(title + dnn + result) 

