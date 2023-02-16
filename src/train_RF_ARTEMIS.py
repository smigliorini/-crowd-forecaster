import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from sklearn.metrics import plot_confusion_matrix

file_name = '../data/ARTEMIS/tuples_context_slimPoI.csv'
trees=10
train = 'raw' #'ctx'
var = 'balanced_class'
out_file = open(f'output_{train}_{trees}_{var}.txt', 'w')

map_class = {49:0 ,61:1, 59:2, 71:3, 54:4, 52:5, 42:6, 58:7, 202:8}
map_class_inv = [49,61,59,71,54,52,42,58,202]

df = pd.read_csv(file_name) 
df['h'] = df.apply(lambda x: int(str(x.istante1)[11:13]) ,axis=1)
df['poi2'] = df['poi2'].map(map_class)
df['poi1'] = df['poi1'].map(map_class)

#49,61,59,71,54,52,42,58,202
for p in range(9):
    begin = time.time()
    file = df[(df['poi1'] == p) & (df['poi2'] != p)]
    out = f'\n------------------------------------------\n{map_class_inv[p]} - size: {len(file)}\n'
    print(out)
    out_file.write(out) 

    if train == 'raw':
        data = file[['month','day','h']]
    else:
        data=file[['month','day','doy','dow','week','hour','holiday','festive',
                   'temperature','humidity','pressure','wind','rain']]

    target = file[['poi2']]

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    n = MinMaxScaler()
    X_train = n.fit_transform(X_train)
    X_test = n.transform(X_test)
 
    weight = compute_class_weight(class_weight = 'balanced', classes=np.unique(Y_train), y=Y_train.values.reshape(-1))
    class_weight = {}
    c=0
    for i in range(9):
        if i != p:
            class_weight[i] = weight[c]
            c += 1 

    RFC = RandomForestClassifier(n_estimators=trees, random_state=0, class_weight=class_weight)
    RFC.fit(X_train, Y_train.values.ravel())

    data_predictions = RFC.predict(X_test)

    fig = plot_confusion_matrix(RFC, X_test, Y_test, display_labels=map_class_inv[:p] + map_class_inv[p+1:], cmap='Greens')
    
    plt.title(f'Random Forest Confusion Matrix - {map_class_inv[p]}')
    plt.savefig(f'./img_{map_class_inv[p]}_{trees}_{train}_{var}.png')
    plt.cla()
    plt.clf()


    #rec = round(recall_score(Y_test, data_predictions, average= 'weighted'), 5)
    #acc =  round(accuracy_score(Y_test, data_predictions), 5)
    #prec = round(precision_score(Y_test, data_predictions, average= 'weighted'), 5)
    all_stats = precision_recall_fscore_support(Y_test, data_predictions, average= 'weighted')
    #output = f'ACC:\t{acc}\nPREC_weighted:\t{prec}\nREC_weighted:\t{rec}\n\n'
    out_stats = f'precision:\t{all_stats[0]}\nrecall:\t{all_stats[1]}\nf1_score:\t{all_stats[2]}\nsupport:\t{all_stats[3]}\n\ntime:{time.time()-begin}\n'

    print(out_stats)
    out_file.write(out_stats)

out_file.close()