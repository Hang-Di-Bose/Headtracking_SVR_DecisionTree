# Overview 
This folder is for running TensorFlow models in Matlab/Simulink. 

You are able to run Python modules in Matlab/Simulink [(see documentation at this link)][1]. So, you are able to load, compile, and predict with TensorFlow models. While it is possible to run this code directly in Matlab, it is not straight forward, and so it is easier to use wrapper functions, which can be found in [tf_matlab.py][2]. Add your own Python functions to that file as you need them.

# Usage
To use this module, copy the folder tf_matlab_module into your Matlab path. Then, to load and train a model, call the following:

    ```matlab
    >> pyversion('/path/to/python/bin/');  % set the desired python version
    >> model = py.tf_matlab_module.tf_matlab.load_model('/path/to/model.h5')
    >> input = py.numpy.array(rand(input_shape)); 
    >> prediction = py.tf_matlab_module.tf_matlab.predict_model(model,input); 
    >> prediction_mat_type = single(prediction)  % double(prediction) is also valid
    ```
    
`py` tells Matlab that the module is a Python module.

`py.tf_matlab_module` tells Matlab the name of the Python module is `tf_matlab_module`.

`py.tf_matlab_module.tf_matlab` tells Matlab the file of interest is named `tf_matlab`.

`py.tf_matlab_module.tf_matlab.predict_model` tells Matlab the function of interest is named `predict_model`.

`py.numpy.array` should be used to convert Matlab arrays to Python arrays if they have more than 2 dimensions. If the array has 1 or 2 dimensions, `py.list(matlab_array)` can be used.

strings do not need to be manually converted from Matlab strings to Python strings, as Matlab correctly converts this type. If you would like to manage the conversion, it is `py.str('string to convert from Matlab to Python')`

`single` converts the return from a Matlab Python object to a Matlab array. `double` is also a valid conversion, but Matlab explicitly calls out `Use single function to convert to a MATLAB array.` when `predict_model` is called. 


[1]: https://www.mathworks.com/help/matlab/call-python-libraries.html?s_tid=CRUX_lftnav
[2]: tf_matlab.py
