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

# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt

def main():
    # tf.compat.v1.enable_v2_behavior()
    print("tensorflow version =",tf.__version__) 
    # get and save params of this run
    params = get_params()

    # dataset = Seq2SeqDataset_copy(
    #     input_path=params.input_dir,
    #     input_window_length_samples =params.input_window_length_samples,
    #     output_window_length_samples=params.output_window_length_samples,
    # )

    # train_dataset = tf.data.Dataset.from_generator((train_x, train_y),output_types=(tf.float64,tf.float64))
    # train_dataset = train_dataset.shuffle(buffer_size=100000)
    # train_dataset = train_dataset.repeat()
    
    datasetD = make_seq_2_seq_dataset(params)

    train_x = datasetD['train']['x']
    train_y = datasetD['train']['y']
    test_x  = datasetD['test']['x']
    test_y  = datasetD['test']['y']
    val_x   = datasetD['val']['x']
    val_y   = datasetD['val']['y']

    train_scenarios = datasetD['train']['scenarios']
    test_scenarios  = datasetD['test']['scenarios']
    val_scenarios   = datasetD['val']['scenarios']
    params.scaleD   = datasetD['scaleD']  # store scaleD in params_out.yml
    
    model = create_model(params)
    model.compile(optimizer=params.optimizer,
                  loss=params.loss,
                  metrics=get_metrics(params))
    
    print(model.summary())


    history = model.fit([train_x],[train_y],
                        batch_size=32,
                        epochs=params.num_epochs, 
                        callbacks=make_callbacks(params),
                        validation_data=([val_x],[val_y]),
                        validation_freq=1)

    # history = model.fit(dataset.train_dataset,
    #                     epochs=params.num_epochs, 
    #                     steps_per_epoch=int(dataset.num_train),
    #                     # callbacks=make_callbacks(params),
    #                     validation_data=dataset.val_dataset,
    #                     validation_steps=int(dataset.num_val),
    #                     validation_freq=1)

    with open(os.path.join(params.output_dir,'history.pickle'),'wb') as f:
        pickle.dump(history.history, f)

    # score = model.evaluate(dataset.test_dataset)
    score = model.evaluate([test_x],[test_y])
    score = [float(s) for s in score]   # convert values in score from np.float to float
    params.score = score                # store score in params_out.yml

    if 'best_checkpoint' in params.callback_list:  # load weights from best checkpoint
        model.load_weights(os.path.join(params.output_dir,"best-checkpoint-weights.h5"))
    elif 'checkpoint' in params.callback_list:
        pass

    save_model(params,model)

    with open(os.path.join(params.output_dir,'params_out.yml'),'w') as f:
        yaml.dump(vars(params), f, default_flow_style=False)

    plot_metrics(params)
    plot_predictions(params,model,test_scenarios)



if __name__=="__main__":
    main()


