import json
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout

def get_model(batch_size, x_dim, y_dim, seq_length, latent_dim_1, latent_dim_2, activation, dropout, optimizer):
    X = Input(batch_shape=(batch_size, seq_length, x_dim), name='X')
    Xf = Flatten()(X)
    Z1 = Dense(latent_dim_1, activation=activation, name='Z')(Xf)
    if dropout > 0.0:
        Z1d = Dropout(dropout)(Z1)
        Z2 = Dense(latent_dim_2, activation=activation, name='Z2')(Z1d)
    else:
        Z2 = Dense(latent_dim_2, activation=activation, name='Z2')(Z1)
    if dropout > 0.0:
        Z2d = Dropout(dropout)(Z2)
        Y = Dense(y_dim, activation='softmax', name='Y')(Z2d)
    else:
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
    if 'latent_dim_1' not in margs: # older version
        margs['latent_dim_1'] = margs['latent_dim']
        margs['latent_dim_2'] = margs['latent_dim']
        margs['activation'] = 'relu'
        margs['dropout'] = 0.0
    if 'add_beats' not in margs: # old version
        margs['add_beats'] = False
        margs['add_holds'] = False
    model = get_model(batch_size, margs['x_dim'], margs['y_dim'], margs['seq_length'], margs['latent_dim_1'], margs['latent_dim_2'], margs['activation'], margs['dropout'], margs['optimizer'])
    model.load_weights(model_file)
    return model, margs
