import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def make_callbacks(params):
    checkpoints = []
    if 'checkpoint' in params.callback_list:
        checkpoints.append(_make_model_checkpoint_cb(params))

    if 'best_checkpoint' in params.callback_list:
        checkpoints.append(_make_best_model_checkpoint_cb(params))
    
    if 'early' in params.callback_list or 'early_stopping' in params.callback_list:
        checkpoints.append(_make_early_stopping_cb(params))

    if 'reduce_lr' in params.callback_list:
        checkpoints.append(_make_reduce_lr_cb(params))

    if 'tensorboard' in params.callback_list:
        checkpoints.append(_make_tensorboard_cb(params))
    return checkpoints 

def _make_model_checkpoint_cb(params):
    filename = os.path.join(params.output_dir,"checkpoint-weights-{epoch:02d}.h5")
    checkpoint = ModelCheckpoint(filename, 
                                monitor=params.checkpoint_metric,
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                save_freq='epoch')
    return checkpoint

def _make_best_model_checkpoint_cb(params):
    filename = os.path.join(params.output_dir,"best-checkpoint-weights.h5")
    checkpoint = ModelCheckpoint(filename, 
                                monitor=params.best_checkpoint_metric,
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                save_freq='epoch')
    return checkpoint

def _make_early_stopping_cb(params):
    early = EarlyStopping(monitor=params.early_stopping_metric,
                            min_delta=0,
                            patience=params.early_stopping_patience,
                            verbose=1,
                            mode='auto')
    return early

def _make_reduce_lr_cb(params):
    reduce_lr = ReduceLROnPlateau(monitor=params.reduce_lr_metric,
                                factor=params.reduce_lr_factor,
                                patience=params.reduce_lr_patience,
                                cooldown=0,
                                verbose=1,
                                mode ='auto')
    return reduce_lr

def _make_tensorboard_cb(params):
    log_dir=os.path.join(params.output_dir,'logs/')
    log_dir=os.path.relpath(path=log_dir)
    os.mkdir(log_dir)
    cb = TensorBoard(log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None,
                    update_freq='epoch')
    return cb
