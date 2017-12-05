import json
import argparse
import numpy as np
from utils.model_utils import get_callbacks, save_model_in_pieces
from utils import load_chorales
from model import get_model

def train(args):
    P = load_chorales.load(args.train_file, args.voice_num, args.seq_length, args.batch_size, voices_to_zero=args.voices_to_zero, use_beats=args.use_beats)
    args.x_dim = P['x_train'].shape[-1]
    args.y_dim = P['y_train'].shape[-1]
    print "Training X with size {} to predict Y with size {}".format(P['x_train'].shape, P['y_train'].shape)
    callbacks = get_callbacks(args, patience=args.patience)

    model = get_model(args.batch_size, args.x_dim, args.y_dim,
        args.seq_length, args.latent_dim_1, args.latent_dim_2,
        args.activation, args.dropout, args.optimizer)

    save_model_in_pieces(model, args)
    history = model.fit(P['x_train'], P['y_train'],
            shuffle=True,
            verbose=0,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            validation_data=(P['x_valid'], P['y_valid']))    
    best_ind = np.argmin([x for i,x in enumerate(history.history['val_loss'])])
    best_loss = {k: history.history[k][best_ind] for k in history.history}
    return model, best_loss

if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--batch_size', type=int, default=200,
        help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
        help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
        help='number of epochs')
    parser.add_argument('--voice_num', type=int,
        default=0, choices=range(4), 
        help='voice number to predict (0 = soprano, ..., 4 = bass)')
    parser.add_argument('--voices_to_zero', type=int,
        default=0, choices=range(4), nargs='+',
        help='voice number to predict (0 = soprano, ..., 4 = bass)')
    parser.add_argument("--use_beats", action="store_true", 
        help="include beat info in X")
    parser.add_argument('--latent_dim_1', type=int, default=100,
        help='latent dim 1')
    parser.add_argument('--latent_dim_2', type=int, default=10,
        help='latent dim 2')
    parser.add_argument('--seq_length', type=int, default=4,
        help='latent dim')
    parser.add_argument('--activation', type=str, default='relu',
        help='activation function')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='dropout (proportion)')
    parser.add_argument('--patience', type=int, default=5,
        help='# of epochs, for early stopping')
    parser.add_argument('--log_dir', type=str,
        default='../data/logs', help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str,
        default='../data/models',
        help='basedir for saving model weights')
    parser.add_argument('--train_file', type=str,
        default='../data/input/JSB Chorales_parts.pickle',
        help='file of training data (.pickle)')
    args = parser.parse_args()
    print train(args)[1]
