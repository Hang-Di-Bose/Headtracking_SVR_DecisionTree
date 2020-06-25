import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import Metric

def get_metrics(params):
    metricsL = []
    if 'mse' in params.eval_metrics or 'mean_squared_error' in params.eval_metrics:
        metricsL.append(tf.keras.metrics.MeanSquaredError())
        pass
    if 'acc' in params.eval_metrics or 'accuracy' in params.eval_metrics:
        metricsL.append(tf.keras.metrics.Accuracy())
        pass
    if 'azimuth_mse'   in params.eval_metrics or 'amse' in params.eval_metrics or 'azimuth_mean_squared_error' in params.eval_metrics:
        metricsL.append(AzimuthMeanSquaredError())
    if 'elevation_mse' in params.eval_metrics or 'emse' in params.eval_metrics or 'elevation_mean_squared_error' in params.eval_metrics:
        metricsL.append(ElevationMeanSquaredError())
    if 'timestep_mse' in params.eval_metrics or 'time_step_mse' in params.eval_metrics or 'tsmse' in params.eval_metrics or 'timestep_mean_squared_error' in params.eval_metrics or 'time_step_mean_squared_error' in params.eval_metrics:
        for i in range(params.output_window_length_samples):
            metricsL.append(TimeStepMeanSquaredError(timestep=i,output_timesteps=params.output_window_length_samples)) 
    return metricsL

class AzimuthMeanSquaredError(Metric):
    """
    Metric for Mean Squared Error in Azimuth (rot_z)

    Attributes 
    ----------
    axes: [int]
        index(es) indicating axis(axes) have azimuth values (rot_z). default [2]
    mse: float
        cumulative mean squared error of epoch
    n: int
        cumulative number of samples in epoch
    """
    def __init__(self, name='azimuth_mean_squared_error', azimuth_axes=[2], **kwargs):
        super(AzimuthMeanSquaredError, self).__init__(name=name, **kwargs)
        self.axes = azimuth_axes 
        self.mse  = self.add_weight(name='mse', initializer=Constant(float("inf")))  # initialize to infinity
        self.n    = self.add_weight(name='n',   initializer=Constant(0),    dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred,tf.float32)

        # get only the axes of interest
        y_true = tf.gather(y_true,self.axes)
        y_pred = tf.gather(y_pred,self.axes)

        # do the maths
        values = tf.math.squared_difference(y_true,y_pred) 
        values = tf.cast(values, tf.float32) 

        # multiply previous mse by previous count of samples
        mse = tf.cast(self.mse, tf.float32)
        mse = tf.multiply(mse, tf.cast(self.n,tf.float32) )

        # update the number of samples
        n = tf.shape(y_true)
        n = tf.math.reduce_prod(n)  # product of all values of shape
        n = tf.cast(n, tf.int32)
        self.n.assign_add(n) 

        # add current squared differences to previous squared differences and then divide by total number of samples
        mse = tf.math.add(mse, tf.math.reduce_sum(values) )
        mse = tf.math.divide(mse, tf.cast(self.n,tf.float32) )
        mse = tf.cast(mse, tf.float32) 
        self.mse.assign(mse)

    def result(self):
        return self.mse

class ElevationMeanSquaredError(Metric):
    """
    Metric for Mean Squared Error in Elevation (rot_x,rot_y,rot_w)

    Attributes 
    ----------
    axis: [int]
        index(es) indicating axis(axes) have elevation values (rot_x,rot_y,rot_w). default [0,1,3]
    mse: float
        cumulative mean squared error of epoch
    n: int
        cumulative number of samples in epoch
    """
    def __init__(self, name='elevation_mean_squared_error', elevation_axes=[0,1,3], **kwargs):
        super(ElevationMeanSquaredError, self).__init__(name=name, **kwargs)
        self.axes = elevation_axes 
        self.mse  = self.add_weight(name='mse', initializer=Constant(float("inf")))  
        self.n    = self.add_weight(name='n',   initializer=Constant(0),    dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred,tf.float32)

        # get only the axes of interest
        y_true = tf.gather(y_true,self.axes)
        y_pred = tf.gather(y_pred,self.axes)

        # do the maths
        values = tf.math.squared_difference(y_true,y_pred) 
        values = tf.cast(values, tf.float32) 

        # multiply previous mse by previous count of samples
        mse = tf.cast(self.mse, tf.float32)
        mse = tf.multiply(mse, tf.cast(self.n,tf.float32) )

        # update the number of samples
        n = tf.shape(y_true)
        n = tf.math.reduce_prod(n)  # product of all values of shape
        n = tf.cast(n, tf.int32)
        self.n.assign_add(n) 

        # add current squared differences to previous squared differences and then divide by total number of samples
        mse = tf.math.add(mse, tf.math.reduce_sum(values) )
        mse = tf.math.divide(mse, tf.cast(self.n,tf.float32) )
        mse = tf.cast(mse, tf.float32) 
        self.mse.assign(mse)


    def result(self):
        return self.mse

class TimeStepMeanSquaredError(Metric):
    """
    Metric for Mean Squared Error in at given timestep

    Attributes 
    ----------
    timestep: int
        index indicating timestep of interest. default 0
    output_timesteps: int
        number of output timesteps (samples). default 0
    nd: [int]
        array for the argument to gather_nd. takes the following form:
        [[0, timestep], [1,timestep], ..., [output_timesteps-1,timestep]]
    mse: float
        cumulative mean squared error of epoch
    n: int
        cumulative number of samples in epoch
    """
    def __init__(self, name='time_step_mean_squared_error', timestep=0, output_timesteps=0, **kwargs):
        super(TimeStepMeanSquaredError, self).__init__(name='time_step_'+str(timestep)+'_mean_squared_error', **kwargs)
        r = tf.range(output_timesteps)
        s = tf.constant([timestep],shape=(output_timesteps,))
        self.nd = tf.stack([r,s],axis=1)
        self.mse = self.add_weight(name='mse', initializer=Constant(float("inf")))  
        self.n   = self.add_weight(name='n',   initializer=Constant(0),    dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred,tf.float32)

        # get only the timestep of interest
        y_true = tf.gather_nd(y_true,self.nd) 
        y_pred = tf.gather_nd(y_pred,self.nd) 

        # do the maths
        values = tf.math.squared_difference(y_true,y_pred) 
        values = tf.cast(values, tf.float32) 

        # multiply previous mse by previous count of samples
        mse = tf.cast(self.mse, tf.float32)
        mse = tf.multiply(mse, tf.cast(self.n,tf.float32) )

        # update the number of samples
        n = tf.shape(y_true)
        n = tf.math.reduce_prod(n)  # product of all values of shape
        n = tf.cast(n, tf.int32)
        self.n.assign_add(n) 

        # add current squared differences to previous squared differences and then divide by total number of samples
        mse = tf.math.add(mse, tf.math.reduce_sum(values) )
        mse = tf.math.divide(mse, tf.cast(self.n,tf.float32) )
        mse = tf.cast(mse, tf.float32) 
        self.mse.assign(mse)


    def result(self):
        return self.mse
