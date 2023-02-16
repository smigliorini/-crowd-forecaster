import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


def build_and_compile_model(norm, nodes, drop):
    model = keras.Sequential([
        norm,
        layers.Dense(nodes, activation='relu'),
        #layers.Dropout(drop),
        layers.Dense(nodes, activation='relu'),
        #layers.Dropout(drop),
        layers.Dense(9, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

def plot_loss(history, title, p, nodes, train, drop):
    plt.cla()
    plt.clf()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./img_{p}_{nodes}_{drop}_{train}.png')

class estimator:
    estimator_type = ''
    classes_=[]
    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
        self.classes_ = classes
    def predict(self, X):
        y_prob= self.model.predict(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred


map_class = {49:0 ,61:1, 59:2, 71:3, 54:4, 52:5, 42:6, 58:7, 202:8}
map_class_inv = [49,61,59,71,54,52,42,58,202]

nodes = 16
train = 'ctx' #'raw'
drop=None

for p in range(9):#[49,61,59,71,54,52,42,58,202]:

    #file_name = f'../data/ARTEMIS/tuples_mapped.csv'# change filename to train on all POIs 
    file_name = f'../data/ARTEMIS/tuples_mapped_{p}.csv'
    df = pd.read_csv(file_name)

    out_file_poi = open(f'./output_{map_class_inv[p]}_{nodes}_{drop}_{train}.txt', 'w')

    begin = time.time()

    file = df[(df['poi1'] == p) & (df['poi2'] != p)]

    out = f'\n------------------------------------------\nTraining PoI: {map_class_inv[p]}, size: {len(file)}\n'
    print(out)
    out_file_poi.write(out)
    out_file_poi.write(f'{map_class_inv[p]}\n')

    if train == 'raw':
        file = file[['poi2','month','day','h']]
    else:
        file = file[['poi2','month','day','doy','dow','week','hour','holiday','festive',
                     'temperature','humidity','pressure','wind','rain']]

    dataset = file.copy()
    dataset.tail()

    dataset.isna().sum() 

    features = dataset.copy()
    labels = features.pop('poi2')

    #split train and test mantaining the same distribution of classes
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=15, stratify=labels)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    normalizer.mean.numpy()

    print(normalizer)

    dnn_model = build_and_compile_model(normalizer, nodes, drop)

    history = dnn_model.fit(
                train_features,
                train_labels,
                epochs=200,
                verbose=1,
                validation_split = 0.2,
            )

    #dnn_model.save(f'./models/model_{p}_{nodes}_{drop}_{epoch}_{train}')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    test_results = {}
    test_results = dnn_model.evaluate(test_features, test_labels, verbose=1)

    print(dnn_model.predict(test_features).shape)

    classifier = estimator(dnn_model, map_class_inv[:p] + map_class_inv[p+1:])

    fig = plot_confusion_matrix(estimator=classifier, X=test_features, y_true=test_labels, display_labels=map_class_inv[:p] + map_class_inv[p+1:], cmap='Greens')

    plt.title(f'DNN Confusion Matrix - {map_class_inv[p]} - DNN {train}')
    plt.savefig(f'./cm_{map_class_inv[p]}_{nodes}_{drop}_{train}.png')
    plt.cla()
    plt.clf()

    y_pred = dnn_model.predict(test_features)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(test_labels, y_pred_bool))

    out_file_poi.write(str(classification_report(test_labels, y_pred_bool)))

    img_title = f'{map_class_inv[p]}-{test_results}'

    out_file_poi.write(f'\ntime:{time.time()-begin}\n')
    out_file_poi.close()
    
    plot_loss(history, img_title, map_class_inv[p], nodes, train, drop)