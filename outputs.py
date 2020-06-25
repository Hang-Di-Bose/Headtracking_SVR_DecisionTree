import json
import os
import pickle
import random 

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def save_model(params,model):
    output_dir = params.output_dir
    model_file_name   = 'out_model_'         + params.timestamp + '.h5'
    weights_file_name = 'out_model_weights_' + params.timestamp + '.h5'
    json_file_name    = 'out_model_json_'    + params.timestamp + '.json'
    model.save(os.path.join(params.output_dir,model_file_name),save_format='h5')
    model.save_weights(os.path.join(params.output_dir,weights_file_name),save_format='h5')
    with open(os.path.join(params.output_dir,json_file_name), 'w') as f:
        json.dump(model.to_json(),f) 

def plot_predictions(params,model,scenarios,num_scenarios=-1,show=False):
    # print('plot predictions')
    # pass
    # """
    output_dir = params.output_dir
    # import ipdb; ipdb.set_trace()
    if num_scenarios == -1:  # use all the scenarios
        num_scenarios = len(scenarios)
    elif num_scenarios < 1: # it's a percent, not a number
        num_scenarios = int(np.ceil(len(scenarios)*num_scenarios))
    else:
        num_scenarios = np.min([num_scenarios,len(scenarios)])
    print('num_scenarios =',num_scenarios)
    selected_scenarios = random.sample(scenarios, num_scenarios)
    for i in range(len(selected_scenarios)):
        print('scenario',i)
        scenario = selected_scenarios[i] 
        x = np.arange(scenario.shape[0]) / 1000 * params.sample_rate    # get time
        plt.figure()

        # plot truth 
        lines = plt.plot(x,scenario)
        plt.legend(lines, ['true rot_x','true rot_y','true rot_z','true rot_w'])
        
        # generate and plot predictions
        j = 0
        while j <= scenario.shape[0] - params.input_window_length_samples - params.output_window_length_samples: 
            window = scenario[j:j+params.input_window_length_samples,:] 
            min_ = np.min(window) 
            max_ = np.max(window) 
            # min_max scale each signal
            # window  = (window  - min_) / (max_ - min_) 
            # out_window = (out_window - min_) / (max_ - min_) 
            window = window.reshape((-1)) 
            x = np.arange(j + params.input_window_length_samples,j + params.input_window_length_samples + params.output_window_length_samples) / 1000 * params.sample_rate 
            prediction = model.predict([[window]]) 
            prediction = prediction[0] # * (max_ - min_) + min_ 
            plt.plot(x,prediction) 
            j += params.output_window_length_samples 
            pass 
        
        # plt.show()
        plt.savefig(os.path.join(output_dir,'test_scenario_'+str(i)))
    
    # if show:
    #     plt.show()
    # """

def plot_metrics(params):
    output_dir = params.output_dir
    # """
    with open(os.path.join(output_dir,'history.pickle'), 'rb') as f:
        history = pickle.load(f)
        train_loss = history['loss']
        val_loss   = history['val_loss']
        epochs     = range(len(train_loss))

        fig = plt.figure() 
        plt.plot(epochs, train_loss, label='training') 
        plt.plot(epochs, val_loss,   label='validation') 
        plt.title('loss: ' + params.loss) 
        plt.legend() 
        fig.savefig(os.path.join(output_dir,'history_loss.png')) 

        if 'mean_squared_error' in params.eval_metrics or 'mse' in params.eval_metrics:
            train_mse = history['mean_squared_error']
            val_mse   = history['val_mean_squared_error']
            epochs     = range(len(train_mse))

            fig = plt.figure() 
            plt.plot(epochs, train_mse, label='training') 
            plt.plot(epochs, val_mse,   label='validation') 
            plt.title('mean squared error') 
            plt.legend() 
            fig.savefig(os.path.join(output_dir,'history_mse.png'))       
    # """ 
