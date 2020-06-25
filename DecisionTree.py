import numpy as np
import json
import os
import pickle
import random
import sys
import yaml
import numpy as np
import tensorflow as tf
from callbacks  import make_callbacks
from datasets   import make_seq_2_seq_dataset, Seq2SeqDataset, Seq2SeqDataset_copy
from metrics    import get_metrics
from models     import create_model
from outputs    import plot_metrics, plot_predictions, save_model
from params     import get_params
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector, Reshape, TimeDistributed
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics

def main():
    start=time.clock()
    params=get_params()

    datasetD = make_seq_2_seq_dataset(params)
    train_x = datasetD['train']['x']
    train_y = datasetD['train']['y']
    test_x = datasetD['test']['x']
    test_y = datasetD['test']['y']


    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples
    train_X=np.array(train_x,dtype=float)
    train_Y=np.array(train_y,dtype=float)
    nsamples, nx = train_X.shape
    print(nsamples)
    print(nx)
    train_Y=np.reshape(train_Y,(nsamples,output_window_samps*num_signals))

    test_X=np.array(test_x,dtype=float)
    print(test_X.shape)
    test_Y=np.array(test_y,dtype=float)
    test_sample,test_nx=test_X.shape
    test_Y=np.reshape(test_Y,(test_sample,output_window_samps*num_signals))
    print(test_Y.shape)

    model_decision=DecisionTreeRegressor(random_state=0)
    model_decision.fit(train_X, train_Y)
    f= open('C:/Users/HD1047208/OneDrive - Bose Corporation/Desktop/data/Hang_decisionTree.pickle','wb')
    pickle.dump(model_decision,f)
    f.close()
    result_decision=model_decision.predict(test_X)
    score_decision=metrics.mean_squared_error(test_Y,result_decision)
    print(result_decision.shape)
    print(score_decision)
    plt.figure(1)
    plt.plot(np.arange(len(result_decision)), result_decision[:,8], 'r-', label='predict value_x0')
    plt.plot(np.arange(len(result_decision)), result_decision[:, 9], 'b-', label='predict value_y0')
    plt.plot(np.arange(len(result_decision)), result_decision[:, 10], 'g-', label='predict value_z0')
    plt.plot(np.arange(len(result_decision)), result_decision[:, 11], 'y-', label='predict value_w0')
    plt.plot(np.arange(len(result_decision)), test_Y[:, 8], 'k-', label='test value_x0')
    plt.plot(np.arange(len(result_decision)), test_Y[:, 9], 'm-', label='test value_y0')
    plt.plot(np.arange(len(result_decision)), test_Y[:, 10], 'c-', label='test value_z0')
    plt.plot(np.arange(len(result_decision)), test_Y[:, 11], 'k-', label='test value_w0')
    plt.xlabel('length')
    plt.ylabel('result')
    plt.title('decision_tree prediction')
    plt.legend()
    plt.show()
   

    print(time.clock()-start)










if __name__=="__main__":
    main()

