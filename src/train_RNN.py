import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Normalization

file_name = '../data/export_61.csv'

nodes = 256
dropout=0.6
epochs=300
batch_size=32
validation_split=0.2
normalization=1
time_steps = 3

def build_and_compile_model_rnn(first_layers_nodes, second_layers_nodes, third_layers_nodes, dropout, X_train):
    return_sequences=False
    if second_layers_nodes>0:
        return_sequences=True
    model = Sequential()
    model.add(LSTM(first_layers_nodes, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=return_sequences))
    model.add(Dropout(dropout))
    if second_layers_nodes>0:
        if third_layers_nodes<=0:
            return_sequences=False

        model.add(LSTM(second_layers_nodes, return_sequences=return_sequences))
        model.add(Dropout(dropout))
    if third_layers_nodes>0:
        model.add(LSTM(third_layers_nodes))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mape'])

    return model


def plot_loss(history, title):
    plt.cla()
    plt.clf()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('Error [Presenze]')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    df = pd.read_csv(file_name)
    #df = df[['presenze','poi', 'month', 'day', 'doy', 'dow', 'week', 'hour', 'holiday', 'festive', 'temperature', 'rain','wind', 'humidity', 'pressure']]

    dataset = df.copy()
    dataset.tail()

    dataset.isna().sum() 
  
    X = df[[ 'month', 'day', 'doy', 'dow', 'week', 'hour', 'holiday', 'festive', 'temperature', 'rain','wind', 'humidity', 'pressure']].values
    y = df['presenze'].values
        
    #normalization 
    if normalization!=0:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    #reshaping into 3-dim array for LSTM
    n_features = X.shape[1]
    X_reshaped = []
    for i in range(time_steps, len(X)+1):
        X_reshaped.append(X[i-time_steps:i, :])
    X_reshaped = np.array(X_reshaped)
    y_reshaped = y[time_steps-1:]

    #test and train split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2)
    normalizer = Normalization(axis=2)
    normalizer.adapt(np.array(X_train))

    # build the model
    rnn_model = build_and_compile_model_rnn(first_layers_nodes = nodes, 
                                            second_layers_nodes = nodes, 
                                            third_layers_nodes=0, 
                                            dropout=dropout,
                                            X_train = X_train)

    # train model
    history = rnn_model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split, 
        verbose=1)

    #rnn_model.save(f'model_rnn_{nodes}_{dropout}_{epochs}')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    print(hist.tail())

    test_results = {}
    test_results = rnn_model.evaluate(
        X_test, y_test)
      
    img_title = '{} - {}'.format(file_name, test_results)
    plot_loss(history, img_title)

    title = f'\n---------------result--------------------\nFile name: {file_name}\n'
    dnn = f'nodes: {nodes} \t dropout: {dropout} \t epochs: {epochs}\n'
    result = f'\nMAPE: {round(test_results[1],1)}\n'

    print(title + dnn + result)
