import tensorflow as tf 

def load_model(path):
    # print(type())
    if type(path) == str:
        model = tf.keras.models.load_model(path)
        print('found .h5 file')
        return model
    elif len(path) == 2 and type(path[0]) == str and type(path[1]) == str:
        model = tf.keras.models.model_from_json(path[0])
        model.load_weights(path[1])
        print('found dict with .h5 and .json file')
        return model 
    else:
        print('nothing done!')

def predict_model(model, data):
    return model.predict(data)
