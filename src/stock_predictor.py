from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_train_data(training_set_scaled, trailing_period=30):
    X_train, y_train = [], []
    for i in range(trailing_period, training_set_scaled.size):
        X_train.append(training_set_scaled[i-trailing_period: i])
        y_train.append(training_set_scaled[i])
    # Converting list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

def compile_and_run(X_train, y_train, model, epochs=50, batch_size=64):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=3)
    return history

def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()

def make_predictions(X_test, model):
    y_pred = model.predict(X_test)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    ap = np.ndarray.flatten(test_set)
    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    ax = pdf.plot()

def create_dl_model(X_train, X_test, y_train,  num_layers = 4):
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units = 50, return_sequences = (num_layers > 1), input_shape = (X_train.shape[1], 1)))

    for i in range(2, num_layers+1):
        model.add(LSTM(units = 50, return_sequences = (i < num_layers))) # return_sequences = False for last layer

    # Adding the output layer
    model.add(Dense(units = 1))
    return model

def get_predictions(scaler, X_test, model):
    y_pred = model.predict(X_test)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    return fp

def eval_layered_model(scaler, X_train, X_test, y_train,  num_layers, verbose=True):
    if verbose:
        print('Evaluating LSTM with ' + str(num_layers) + ' layers.')
    dl_model = create_dl_model(X_train, X_test, y_train, num_layers=num_layers)
    if verbose:
        dl_model.summary()
    history = compile_and_run(X_train, y_train, dl_model, epochs=10)
    if verbose:
        plot_metrics(history)
    return get_predictions(scaler, X_test, dl_model)

def predict_ticker(df, ticker, year=2017, trailing_days=30, verbose=True):
    if verbose:
        print('Predicting for: ' + ticker + ' in year ' + str(year) + ' using ' + str(trailing_days) + ' days')
    ticker_df = df[df.Name == ticker]
    tdf = ticker_df[['Date', 'Price']].sort_values('Date')
    training_set = tdf[tdf.Date.dt.year != year].Price.values
    test_set =  tdf[tdf.Date.dt.year == year].Price.values
    if verbose:
    	print("Training set size: ",training_set.size)
    	print("Test set size: ", test_set.size)

    scaler = MinMaxScaler()
    training_set_scaled = scaler.fit_transform(training_set.reshape(-1, 1))
    X_train, y_train = create_train_data(training_set_scaled,trailing_period=trailing_days)
    X_test = []
    inputs = tdf[len(tdf) - len(test_set) - trailing_days:].Price.values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    for i in range(trailing_days, test_set.size+trailing_days): # Range of the number of values in the training dataset
        X_test.append(inputs[i - trailing_days: i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    ap = np.ndarray.flatten(test_set)
    data = {'Actual': ap}
    for num_layers in range(1,3):
        predictions = eval_layered_model(scaler, X_train, X_test, y_train, num_layers,verbose=verbose)
        data['Predicted by '+ str(num_layers) + ' layers'] = predictions

    pdf = pd.DataFrame(data=data)
    sns.set(rc={'figure.figsize':(16,10)})
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Stock Price', fontsize=16)
    plt.title(ticker + ' actual price versus price predicted by LSTM for various numbers of layers', fontsize=16)
    ax = pdf.plot(colormap='tab10',lw=3,alpha=0.5)
    plt.savefig('images/' + ticker +'_LSTM_layers.png')
