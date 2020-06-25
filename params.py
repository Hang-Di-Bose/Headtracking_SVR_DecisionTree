import argparse 
import datetime 
import os 
import sys 
import yaml 

import numpy as np 

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run training for HeadMovementPrediction')
    parser.add_argument('--input_dir',   type=str,   required=True,
                        help='path to input data')
    parser.add_argument('--output_dir', type=str,   required=True,
                        help='')
    parser.add_argument('--output_suffix', type=str,required=False, default='',
                        help='')
    parser.add_argument('--path_to_recording_session',type=str,required=True,
                        help="absolute or relative path to folder containing master branch of RecordingSession. RecordingSession repo is on BoseCorp github, and should be pulled down locally")

    # model arguments
    parser.add_argument('--model_type',         type=str,   default='seq_2_seq_svr',
                        help='the name of the model to train. see models.py for valid model names')
    parser.add_argument('--num_epochs',         type=int,   default=100,
                        help='number of epochs for training model')
    parser.add_argument('--batch_size',         type=int,   default=32,
                        help='batch for training data')
    parser.add_argument("--optimizer",          type=str,   default="adam",
                        help="specify the optimizer for the model")
    parser.add_argument("--base_learning_rate", type=int,   default=0.001,
                        help="specify the base learning rate for the specified optimizer for the model")
    parser.add_argument("--loss",               type=str,   default="categorical_crossentropy",
                        help="specify the loss calculation method for the model. Options [categorical_crossentropy | binary_crossentropy]")
    parser.add_argument("--eval_metrics",       type=str,   default=None,
                        help="a list of the metrics of interest, seperated by commas")

    ## callback arguments
    parser.add_argument("--callback_list",      type=str,   default=None,
                        help="the callbacks to be added. see callbacks.py for available callbacks.")
    parser.add_argument("--early_stopping_patience",    type=int,   default=30,
                        help="patience for the early stopping callback (if specified)")
    parser.add_argument("--reduce_lr_patience",         type=int,   default=30,
                        help="patience for the reduce learning rate callback (if specified)")
    parser.add_argument("--reduce_lr_factor",           type=int,   default=0.2,
                        help="factor for the reduce learning rate callback (if specified)")
    parser.add_argument("--checkpoint_metric",          type=str,   default='val_loss',
                        help="monitor argument for the early stopping callback (if specified)")
    parser.add_argument("--best_checkpoint_metric",     type=str,   default='val_loss',
                        help="monitor argument for the reduce learning rate callback (if specified)")
    parser.add_argument("--early_stopping_metric",      type=str,   default='val_loss',
                        help="monitor argument for the early stopping callback (if specified)")
    parser.add_argument("--reduce_lr_metric",           type=str,   default='val_loss',
                        help="monitor argument for the reduce learning rate callback (if specified)")

    # seq_2_seq_lstm args
    parser.add_argument("--lstm0_units",    type=int,   default=10)
    parser.add_argument("--lstm1_units",    type=int,   default=10,
                        help="in cases with mutliple LSTM layers, the number of cells in the LSTM layer")

    # data parameters
    parser.add_argument("--input_window_length_ms",     type=int,   default=500)
    parser.add_argument("--output_window_length_ms",    type=int,   default=30)
    parser.add_argument("--window_hop_ms",              type=int,   default=30)
    parser.add_argument("--sample_rate",                type=int,   default=100)
    parser.add_argument("--num_signals",                type=int,   default=4)
    parser.add_argument("--write_csv",                  type=str2bool,  default=False)
    parser.add_argument("--normalization",              type=str,   default="baseline",
                        help="baseline is normal; signal is different normalization for each axis; window is min_max normalizatin for windows")

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_params():
    parser = make_argparser()

    # get the timestamp of when you started, and append it to the necessary paths
    parser.timestamp = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")   
    parser.output_dir = parser.output_dir + '_' + parser.timestamp + parser.output_suffix 
    if not os.path.isdir(parser.output_dir):
        os.mkdir(parser.output_dir)
    if parser.write_csv:
        if not os.path.isdir(os.path.join(parser.output_dir,'data')):
            os.mkdir(os.path.join(parser.output_dir,'data'))

    parser.eval_metrics  = parser.eval_metrics.split(',')  if parser.eval_metrics  is not None else [] 
    parser.callback_list = parser.callback_list.split(',') if parser.callback_list is not None else [] 

    sr = parser.sample_rate
    parser.input_window_length_samples  = int(np.floor(parser.input_window_length_ms)  / 1000 * sr)
    parser.output_window_length_samples = int(np.floor(parser.output_window_length_ms) / 1000 * sr)
    parser.window_hop_samples     = int(np.floor(parser.window_hop_ms)     / 1000 * sr)

    parser.data_dir = parser.input_dir

    # save parser struct in params.yml file 
    with open(os.path.join(parser.output_dir,'params.yml'),'w') as f:
        yaml.dump(vars(parser), f, default_flow_style=False)
        
    # add path_to_recording_session to os.path 
    rsDir = parser.path_to_recording_session 
    if rsDir not in sys.path: 
        sys.path.insert(1,rsDir) 
        sys.path.insert(2,os.path.join(rsDir,'RecordingSession')) 

    #print(parser)
    return parser 
