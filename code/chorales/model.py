import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten

def get_model(batch_size, x_dim, y_dim, seq_length, latent_dim, optimizer):
    X = Input(batch_shape=(batch_size, seq_length, x_dim), name='X')
    Z = Dense(latent_dim, activation='relu', name='Z')(X)
    Zf = Flatten()(Z)
    Z2 = Dense(latent_dim, activation='relu', name='Z2')(Zf)
    Y = Dense(y_dim, activation='softmax', name='Y')(Z2)
    mdl = Model(X, Y)
    mdl.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return mdl
