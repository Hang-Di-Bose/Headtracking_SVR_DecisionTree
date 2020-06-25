import os 
import re 
import random
import sys 
import yaml 

import numpy as np 
import tensorflow as tf 

from tensorflow.keras import Model 
# from tensorflow.keras.Model  import compile, load_weights, predict
from tensorflow.keras.models import load_model, model_from_json

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 

def plot_predictions_post_process(base_path):
    # pass
    # """
    yml_path = os.path.join(base_path,'params.yml')
    with open(yml_path) as f: 
        params = yaml.safe_load(f)
    timestamp   = params['timestamp'] 
    sample_rate = params['sample_rate']
    in_window_length_samples  = params['input_window_length_samples']
    out_window_length_samples = params['output_window_length_samples']

    model_path      = os.path.join(base_path,'out_model_'         + timestamp + '.h5')
    model_json_path = os.path.join(base_path,'out_model_json_'    + timestamp + '.json')
    weights_path    = os.path.join(base_path,'out_model_weights_' + timestamp + '.h5')

    if os.path.exists(weights_path) and os.path.exists(model_json_path):
        model = model_from_json(model_json_path)
        model.load_weights(weights_path)
    else:
        model = load_model(model_path)
    
    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    if params['write_csv']:
        data_path   = os.path.join(base_path,'data')
        # val_files = 
        test_files  = [f for f in os.listdir(data_path) if f.endswith('test.npy')]
        print(len(test_files),'test_files found')
        # train_files = 
        # test_files = random.sample(test_files,np.min([len(test_files),10]))
        # test_arrays = [] 
        for test_file in test_files: 
            test_array = np.load(os.path.join(data_path,test_file)) 
            print(test_file,'\tshape =',test_array.shape) 
            if test_array.shape[0] < 10000:
                x = np.arange(test_array.shape[0]) / 1000 * sample_rate 

                plt.figure() 
                plt.title(test_file) 

                lines = plt.plot(x,test_array) 
                plt.legend(lines, ['true rot_x','true rot_y','true rot_z','true rot_w']) 
                j = 0
                while j <= test_array.shape[0] - in_window_length_samples - out_window_length_samples:
                    window = test_array[j:j+in_window_length_samples,:] 
                    window = window.reshape((-1)) 
                    x = np.arange(j + in_window_length_samples, j + in_window_length_samples + out_window_length_samples) / 1000 * sample_rate 
                    prediction = model.predict([[window]]) 
                    plt.plot(x,prediction[0]) 
                    j += out_window_length_samples 

        plt.show()
