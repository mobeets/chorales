import os.path
import argparse
import datetime
import numpy as np
import pandas as pd
from chorales.train import train as train_fcn

# for running in parallel
import multiprocessing
from functools import partial
from contextlib import contextmanager

def get_hyperopts(args):
    opts = {}
    nvs = []
    for (key, val) in vars(args).iteritems():
        if type(val) is list:
            opts[key] = val
            nvs.append(len(val))
    return opts

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def update_args(args, opts, base_name):
    np.random.seed()
    cargs = vars(args)
    name = base_name
    for (key,vals) in opts.iteritems():
        if not args.random:
            val = np.random.choice(vals)
        else:
            if len(vals) == 1:
                vals = [vals[0]]*2
            assert len(vals) == 2, "if random, provide min and max"
            mn, mx = vals
            if mn != 0 and np.log10(mx) - np.log10(mn) > 1:
                val = np.exp(np.random.uniform(np.log(mn), np.log(mx)))
            else:
                if type(mn) is int:
                    # give edges an equal change
                    val = np.min([np.max([np.random.uniform(mn-0.5, mx+0.5), mn]), mx])
                else:
                    val = np.random.uniform(mn, mx)
            if type(mn) is int: # round if necessary
                val = int(np.round(val))
        cargs[key] = val
        name += '-' + key + str(cargs[key])
    cargs['run_name'] = name
    return SimpleNamespace(**cargs)

def write_to_csv(outfile, params, losses, dts):
    if type(losses[0]) is int or type(losses[0]) is float:
        losses = [[l] for l in losses]
    if type(losses[0]) is list:
        losses = [{'loss_{}'.format(i):l for i,l in ls} for ls in losses]    

    rows = []
    for loss, ps, dt in zip(losses, params, dts):
        loss.update(ps)
        loss['timestamp'] = dt
        rows.append(loss)
    pd.DataFrame(rows).to_csv(outfile + '.csv')

def hypersearch(index, args):
    """
    partial grid search on any args with multiple options
    """
    base_name = args.run_name
    if index is not None:
        base_name += '_i={}'.format(index)
    outfile = os.path.join(args.log_dir, base_name)
    opts = get_hyperopts(args)
    losses = []
    params = []
    dts = []
    c = 0
    d = 0
    maxd = 2*args.num_runs # prevent infinite loops
    min_loss = np.inf
    while True:
        d += 1
        if d > maxd:
            break
        if c >= args.num_runs:
            break
        cargs = update_args(args, opts, base_name)
        if 'voices_to_zero' in opts:
            cargs.voices_to_zero = opts['voices_to_zero']
        ps = vars(cargs)
        if ps in params:
            continue

        print '==============================='
        print cargs.run_name
        print '==============================='
        # model, loss = train_fcn(cargs)
        try:
            model, loss = train_fcn(cargs)
        except Exception,e:
            print str(e)
            print "FAILURE"
            continue
        losses.append(loss)
        params.append(ps)
        dts.append(datetime.datetime.now())
        if type(loss) is list:
            loss = loss[0]
        elif type(loss) is dict:
            loss = loss['val_loss']
        if loss < min_loss:
            min_loss = loss
            print "NEW MINIMUM"
        c += 1
        write_to_csv(outfile, params, losses, dts)

    for i in xrange(len(params)):
        print params[i], losses[i]

def concat_csvs(args):
    dfs = []
    for i in range(args.num_threads):
        fnm = args.run_name + '_i={}.csv'.format(i)
        outfile = os.path.join(args.log_dir, fnm)
        dfs.append(pd.read_csv(outfile))
    outfile = os.path.join(args.log_dir, args.run_name + '.csv')
    df = pd.concat(dfs).to_csv(outfile)

if __name__ == '__main__':
    """
    Any argument that you provide multiple options to, this will perform a random hyperparameter search over these values
        NOTE: This script doesn't know what arguments these models actually use, it just provides all of them to the train function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument("--random", action="store_true", 
        help="instead of grid, do random selection between provided range")
    parser.add_argument('--num_runs', type=int, default=128,
        help='number of runs')
    parser.add_argument('--num_threads', type=int, default=1,
        help='number of parallel threads')

    parser.add_argument('--optimizer', type=str, default='adam',
        help='optimizer name') # 'rmsprop', 'nadam'
    parser.add_argument('--patience', type=int, default=5,
        help='# of epochs, for early stopping')
    parser.add_argument('--num_epochs', type=int, default=200,
        help='number of epochs')
    parser.add_argument('--batch_size', nargs='+', type=int,
        default=[50, 100])
    parser.add_argument('--latent_dim_1', nargs='+', type=int,
        default=4)
    parser.add_argument('--latent_dim_2', nargs='+', type=int,
        default=4)
    parser.add_argument('--seq_length', nargs='+', type=int,
        default=1)
    parser.add_argument('--activation', nargs='+', type=str,
        default='relu')
    parser.add_argument('--dropout', nargs='+', type=float,
        default=0.0)

    parser.add_argument('--voice_num', type=int,
        default=0, choices=range(4), 
        help='voice number to predict (0 = soprano, ..., 4 = bass)')
    parser.add_argument('--voices_to_zero', type=int,
        default=0, choices=range(4), nargs='+',
        help='voice number to predict (0 = soprano, ..., 4 = bass)')
    parser.add_argument('--log_dir', type=str, default='../data/logs',
        help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str,
        default='../data/models',
        help='basedir for saving model weights')
    parser.add_argument('--train_file', type=str,
        default='../data/input/JSB Chorales_parts.pickle',
        help='file of training data (.pickle)')
    args = parser.parse_args()

    if args.num_threads == 1:
        hypersearch(index=None, args=args)
    else:
        @contextmanager
        def poolcontext(*args, **kwargs):
            pool = multiprocessing.Pool(*args, **kwargs)
            yield pool
            pool.terminate()
        with poolcontext(processes=args.num_threads) as pool:
            results = pool.map(partial(hypersearch, args=args), range(args.num_threads))
        concat_csvs(args)
