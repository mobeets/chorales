import json
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM

def get_model(batch_size, x_dim, y_dim, seq_length, latent_dim, optimizer):
    X = Input(batch_shape=(batch_size, seq_length, x_dim), name='X')
    
    # Z0 = LSTM(latent_dim, name='Z0')(X)
    # Z2 = Dense(latent_dim, activation='relu', name='Z')(Z0)

    Xf = Flatten()(X)
    Z = Dense(latent_dim, activation='relu', name='Z')(Xf)
    Z2 = Dense(latent_dim, activation='relu', name='Z2')(Z)
    Y = Dense(y_dim, activation='softmax', name='Y')(Z2)
    mdl = Model(X, Y)
    mdl.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return mdl

def load_model(model_file, batch_size=1):
    """
    there's a curently bug in the way keras loads models from .yaml
        that has to do with Lambdas
    so this is a hack for now...
    """
    margs = json.load(open(model_file.replace('.h5', '.json')))
    batch_size = batch_size if batch_size is not None else margs['batch_size']
    model = get_model(batch_size, margs['x_dim'], margs['y_dim'], margs['seq_length'], margs['latent_dim'], margs['optimizer'])
    model.load_weights(model_file)
    return model, margs
