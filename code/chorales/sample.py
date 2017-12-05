import json
import os.path
import argparse
import numpy as np
from utils import load_chorales
from utils.midi_utils import write_song
from model import load_model

def write_sample(X, y, P, run_name, sample_dir, use_holds, margs, postfix=''):
    song = load_chorales.X_and_y_to_song(X, y, margs['voice_num'], P['offsets'], P['ranges'], use_holds)
    if use_holds:
        holds = song[1]
        song = song[0]
    else:
        holds = None
    fnm = run_name + '{}.mid'.format(postfix)
    fnm = os.path.join(sample_dir, fnm)
    write_song(song, fnm, isHalfAsSlow=True, holds=holds)

def generate_sample(model, X, nt, seq_length, y_dim, ranges, voice_num, use_argmax=False):
    Xcur = X[0].copy()
    y_start_ind = ranges[:voice_num].sum() + voice_num
    ys = []
    Xs = []
    for t in xrange(nt):
        ph = model.predict(Xcur[None,:,:], batch_size=1)[0]

        # sample y_t ~ Categorical(ph)
        if use_argmax:
            y_t = np.argmax(ph)
        else:
            y_t = np.random.choice(y_dim, p=ph)
        v_t = np.zeros(y_dim)
        v_t[y_t] = 1.

        # add to X[t]
        ys.append(v_t)
        Xcur[0][y_start_ind:(y_start_ind+y_dim)] = v_t
        Xs.append(Xcur)

        # update history terms
        if t < nt-1:
            # todo
            Xcur = np.vstack([np.zeros((1,Xcur.shape[1])), Xcur])
            Xcur[0] = X[t+1,0] # add next X
            Xcur = Xcur[:-1] # ignore last history
    return np.vstack(ys), np.dstack(Xs).swapaxes(0,2).swapaxes(1,2)

def sample(args):
    model, margs = load_model(args.model_file)
    P = load_chorales.load(margs['train_file'], margs['voice_num'], margs['seq_length'], margs['batch_size'], voices_to_zero=margs['voices_to_zero'], use_beats=margs['use_beats'])
    for i in xrange(args.nsamples):
        ind = 100*i
        X = P['x_valid'][ind:ind+args.sample_length,:,:]
        y = P['y_valid'][ind:ind+args.sample_length]
        yh, _ = generate_sample(model, X, args.sample_length, margs['seq_length'], margs['y_dim'], P['ranges'], margs['voice_num'], args.use_argmax)
        write_sample(X, y, P, args.run_name, args.sample_dir, args.use_holds, margs, '{}_true'.format(i))
        write_sample(X, yh, P, args.run_name, args.sample_dir,args.use_holds, margs, '{}_pred'.format(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('-n', '--nsamples',
        type=int, default=5,
        help='number of samples')
    parser.add_argument('-t', '--sample_length',
        type=int, default=32,
        help='number of samples')
    parser.add_argument("--use_holds", action="store_true", 
        help="X contains note holds")
    parser.add_argument("--use_argmax", action="store_true", 
        help="play argmax note instead of sampling")
    parser.add_argument('--sample_dir', type=str,
        default='../data/output',
        help='basedir for saving samples')
    parser.add_argument('--model_dir', type=str,
        default='../data/models',
        help='basedir for saving model weights')
    parser.add_argument('--model_file', type=str, default=None,
        help='file to load weights')
    args = parser.parse_args()
    sample(args)
