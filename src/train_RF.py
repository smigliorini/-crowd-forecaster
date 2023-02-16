import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

file_name = '../data/export_61.csv'
trees=100
train = 'raw' 
#train = 'context'

if __name__ == "__main__":
    file = pd.read_csv(file_name) 

    if train == 'raw':
        data=file[[ 'month','day','doy','week','hour']] 
    else:
        data=file[['month','day','doy','dow','week','hour','holiday','festive','temperature','humidity','pressure','wind','rain']]

    target = file[[ 'presenze']]

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    n = MinMaxScaler()
    X_train = n.fit_transform(X_train)
    X_test = n.transform(X_test)

    forest_reg = RandomForestRegressor(n_estimators=trees, random_state=0)
    forest_reg.fit(X_train, Y_train.values.ravel())

    data_predictions = forest_reg.predict(X_test)

    mape = round(mean_absolute_percentage_error(Y_test, data_predictions), 5)

    title = '\n---------------result--------------------\nFile name: {}\nData tipe: {}\n'.format(file_name, train)
    output = 'n_estimator: {}\tmape: {}\n'.format(trees, mape)

    print(title+output)
