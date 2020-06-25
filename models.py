import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector, Reshape, TimeDistributed
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing



def create_seq_2_seq_svr(params):
    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples
    units0 = params.lstm0_units





    input = Input(shape=(input_window_samps * num_signals,))
    x = Reshape((input_window_samps, num_signals))(input)
    #x=GridSearchCV(SVR(x,y),param_grid={'kernel':('linear','rbf','sigmoid'),'C':np.logspace(-3, 3, 7),'gamma':np.logspace(-3, 3, 7)})
    x=SVR(kernel='rbf',degree=3)(x)
    model = Model(inputs=input, outputs=x)
    return model


def create_seq_2_seq_lstm(params):
    # from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ "Multiple Parallel Input and Multi-Step Output" example
    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples

    units1 = params.lstm1_units

    input = Input(shape=(input_window_samps * num_signals,))
    x = Reshape((input_window_samps, num_signals))(input)
    x = LSTM(units0, activation='relu')(x)
    x = RepeatVector(output_window_samps)(x)
    x = LSTM(units1, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(num_signals))(x)

    model = Model(inputs=input, outputs=x)
    return model


def create_seq_2_seq_multi_encoder(params):
    # from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ "Multiple Parallel Input and Multi-Step Output" example
    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples
    units0 = params.lstm0_units
    units1 = params.lstm1_units

    input = Input(shape=(input_window_samps * num_signals,))
    x = Reshape((input_window_samps, num_signals))(input)

    encoders = []
    for _ in range(output_window_samps):
        encoders.append(LSTM(units0, activation='relu')(x))

    x = Concatenate(axis=-1)(encoders)
    x = Reshape((output_window_samps, 10))(x)

    x = LSTM(units1, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(num_signals))(x)

    model = Model(inputs=input, outputs=x)
    return model


def create_seq_2_seq_vector(params):
    # from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ "Multiple Parallel Input and Multi-Step Output" example

    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples
    units0 = params.lstm0_units

    input = Input(shape=(input_window_samps * num_signals,))
    x = Reshape((input_window_samps, num_signals))(input)
    x = LSTM(units0, activation='relu')(x)
    x = RepeatVector(output_window_samps)(x)
    x = Dense(output_window_samps * num_signals)(x)
    x = Reshape((output_window_samps, num_signals))(x)

    model = Model(inputs=input, outputs=x)
    return model


def create_seq_2_seq_vector_and_multi_encoder(params):
    # from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ "Multiple Parallel Input and Multi-Step Output" example

    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples
    units0 = params.lstm0_units

    input = Input(shape=(input_window_samps * num_signals,))
    x = Reshape((input_window_samps, num_signals))(input)

    encoders = []
    for _ in range(output_window_samps):
        encoders.append(LSTM(units0, activation='relu')(x))

    x = Concatenate(axis=-1)(encoders)
    x = Reshape((output_window_samps, 10))(x)

    x = Dense(output_window_samps * num_signals)(x)
    x = Reshape((output_window_samps, num_signals))(x)

    model = Model(inputs=input, outputs=x)
    return model


def create_model(params):
    model_type = params.model_type
    if model_type == 'seq_2_seq_svr':
        return create_seq_2_seq_svr(params)
    if model_type == 'seq_2_seq_lstm' or model_type == 'seq_2_seq_encoder':
        return create_seq_2_seq_lstm(params)
    if model_type == 'seq_2_seq_multi_encoder':
        return create_seq_2_seq_multi_encoder(params)
    if model_type == 'seq_2_seq_vector':
        return create_seq_2_seq_vector(params)
    if model_type == 'seq_2_seq_vector_and_multi_encoder':
        return create_seq_2_seq_vector_and_multi_encoder(params)
    else:
        message = 'No model_type = ' + model_type
        assert (message)

