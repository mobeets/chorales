import pickle
import numpy as np
from music21 import key
from keras.utils import to_categorical
from utils.midi_utils import write_song

SILENCE = -1
DNE = -2
HOLD_NOTE = 128

def findOffsetToStandardize(curKey, keyGoal):
    if type(curKey) is str:
        curKey = key.Key(curKey)
    i = intervalToKey(curKey, keyGoal)
    return i.chromatic.semitones, curKey.mode == key.Key(keyGoal).mode

def change_key_of_song(song, curKey, goalKey, const=0):
    offset, modeMatch = findOffsetToStandardize(curKey, goalKey)
    return [[y+offset+const if y!=SILENCE else y for y in x] for x in song]

def standardize_key(train_file, outfile, goalKey='C', goalKeyMin='a'):
    d = pickle.load(open(train_file))
    e = {}
    mns = [60, 53, 48, 36]
    mxs = [81, 74, 69, 64]
    for k in ['train', 'test', 'valid']:
        e[k] = []
        e[k + '_key'] = []
        e[k + '_mode'] = d[k + '_mode']
        for i, song in enumerate(d[k]):
            curKey = d[k + '_key'][i]
            song = change_key_of_song(song, curKey, goalKey)
            X = np.vstack(song)
            if np.any(X.max(axis=0) > mxs):
                song = change_key_of_song(song, curKey, goalKey, const=-12)
            X = np.vstack(song)
            X2 = X.copy()
            X2[X2 == SILENCE] = 1e5
            if np.any(X2.min(axis=0) < mns):
                song = change_key_of_song(song, curKey, goalKey, const=12)
            assert len(e[k]) == i
            e[k].append(song)
            e[k + '_key'] = curKey if d[k + '_mode'][i] else goalKeyMin
    pickle.dump(e, open(outfile, 'w'))

def find_note_ranges(d):
    mns = []
    mxs = []
    for k in ['train', 'test', 'valid']:
        X = np.vstack(d[k])
        X2 = X.copy()
        X2[X2 == SILENCE] = 1e4
        mns.append(X2.min(axis=0))
        mxs.append(X.max(axis=0))
    mns = np.vstack(mns).min(axis=0)
    mxs = np.vstack(mxs).max(axis=0)
    return mns, mxs-mns+1

def infer_delta_from_beats(d):
    vs = []
    for k in ['train', 'test', 'valid']:
        v = np.unique(np.hstack(d[k + '_beats']))
        vs.append(v)
    all_beats = np.unique(np.array(vs))
    ds = np.unique(all_beats - all_beats.astype(int))
    delta = ds[ds > 0].min()
    n_beats = (all_beats/delta).max().astype(int)
    return delta, n_beats

def add_history_to_grid(grid, seq_length, dne=DNE):
    """
    takes [n x d] grid and makes it [n x d x seq_length]
        by adding history terms
    """
    if hasattr(grid, 'shape') and len(grid.shape) == 1:
        grid = grid[:,None]
    xs = np.dstack([np.roll(grid, i, axis=0) for i in xrange(seq_length)])
    xs[:(seq_length-1),:,1:] = dne # dne
    return xs

def add_history_and_stack_grids(grids, seq_length):
    xss = [add_history_to_grid(grid, seq_length) for grid in grids]
    X = np.vstack(xss) # [n x d x seq_length]
    X = X.swapaxes(1,2) # -> [n x seq_length x d]
    return X

def trim_for_batch_size(X, batch_size):
    # ignore ends to be divisible by batch size
    n = (len(X) / batch_size)*batch_size
    return X[:n]

def make_hist_and_offset(d, seq_length, batch_size, use_beats=False):
    """
    build history terms and subtract min note number
    """
    offsets, ranges = find_note_ranges(d)
    D = {'offsets': offsets, 'ranges': ranges}

    keys = ['train', 'test', 'valid']
    if use_beats:
        keys += [k + '_beats' for k in ['train', 'test', 'valid']]
        D['beat_delta'], D['n_beats'] = infer_delta_from_beats(d)

    for k in keys:
        # add history terms
        X = add_history_and_stack_grids(d[k], seq_length)

        if 'beats' not in k:
            # shift so everything starts at 0
            ix1 = (X == SILENCE)
            ix2 = (X == DNE)
            X -= (offsets-2) # now all start at 2
            X[ix1] = 1 # -1 -> 1 for silence
            X[ix2] = 0 # -2 -> 0 for dne

        D[k] = trim_for_batch_size(X, batch_size)
    return D

def X_to_onehot(X, num_notes):
    """
    X: [n x k x d]
    num_notes: [d x 1]
    returns [n x k x d*sum(num_notes)]
    """
    xs = []
    for i in xrange(X.shape[-1]):
        xcs = []
        for j in xrange(X.shape[1]):
            xc = to_categorical(X[:,j,i], num_classes=num_notes[i])
            xcs.append(xc)
        xc = np.dstack(xcs).swapaxes(1,2)
        xc = xc[:,:,1:] # remove dne col -> all zeros
        xs.append(xc)
    return np.dstack(xs) # [n x seq_length x nnotes]

def y_to_onehot(y, num_notes):
    y = to_categorical(y, num_classes=num_notes+2)
    return y[:,1:] # remove dne col -> all zeros

def make_X_and_y(D, yind, use_beats=False, inds_to_zero=None):
    """
    separate X from y
    """
    for k in ['train', 'test', 'valid']:
        X = D[k]

        # make y
        y = X[:,0,yind].copy()
        y = y_to_onehot(y, D['ranges'][yind])

        # make X, with some inds (and yind) zeroed out
        if inds_to_zero is not None:
            X[:,0,inds_to_zero] = 0 # 0 -> dne
        X[:,0,yind] = 0 # 0 -> dne
        X = X_to_onehot(X, D['ranges']+2)

        if use_beats:
            # make one-hot; append to X
            Z = D[k + '_beats'] # [n x seq_length x 1?]
            Z[Z == DNE] = 0 # no zero beat anyway
            Z = Z/D['beat_delta']
            Z = X_to_onehot(Z, [D['n_beats']+1])
            X = np.dstack([X, Z])

        D['x_' + k] = X
        D['y_' + k] = y
    return D

def onehot_to_y(y, offset):
    row, ind = np.where(y)
    if len(row) == 0:
        # zero'd out part
        return SILENCE*np.ones(len(y))
    assert len(ind) == len(y)
    ix0 = ind == 0
    ind += (offset-1)
    ind[ix0] = SILENCE
    return ind

def X_and_y_to_song(X, y, yind, offsets, ranges, use_holds):
    x = X[:,0,:] # drop history
    x = x[:,:ranges.sum()+4]
    voice_inds = np.hstack([i*np.ones(r+1) for i,r in enumerate(ranges)])
    song = []
    holds = []
    for i,offset in enumerate(offsets):
        # each voice is a matrix of one-hot vectors
        if i == yind:
            part = y
        else:
            part = x[:,voice_inds == i]
        # convert one-hot vectors to note indices
        inds = onehot_to_y(part, offset)
        if inds[0] == HOLD_NOTE:
            # if first note is hold, ignore
            inds[0] = SILENCE
        is_hold = (inds == HOLD_NOTE)        
        while (inds == HOLD_NOTE).sum() > 0:
            hold_t = np.where(inds == HOLD_NOTE)[0] # hold note times
            inds[hold_t] = inds[hold_t-1] # set to previous note
        assert inds.max() < HOLD_NOTE
        if (inds != SILENCE).sum() > 0:
            print inds
            print is_hold.astype(int)
            print i, offset, inds.min(), inds.max(), is_hold.sum()
        song.append(inds)
        holds.append(is_hold)
    if use_holds:
        return np.vstack(song).T, np.vstack(holds).T # [n x 4]
    else:
        return np.vstack(song).T # [n x 4]

def test_songs_are_invertible(D, d, voice_num):
    song0 = np.vstack([np.vstack(x) for x in d['train']])
    song1 = X_and_y_to_song(D['x_train'], D['y_train'], voice_num, D['offsets'], D['ranges'], use_holds=False)
    # must be false to check this, I think
    assert (song0 - song1).sum() == 0

def write_songs(songs):
    for i, song in enumerate(songs):
        fnm = '../data/output/sample_{}.mid'.format(i)
        write_song(song, fnm, isHalfAsSlow=True)

def load(train_file='../data/input/JSB Chorales_parts.pickle', voice_num=0, seq_length=8, batch_size=1, voices_to_zero=None, use_beats=False):
    """
    load data by parts
    where you predict one part using the other parts
        plus the sequence history
    """
    d = pickle.load(open(train_file))
    D = make_hist_and_offset(d, seq_length, batch_size, use_beats)
    D = make_X_and_y(D, voice_num, use_beats, inds_to_zero=voices_to_zero)
    # test_songs_are_invertible(D, d, voice_num)
    # write_songs(d['train'][:10])

    # X = D['x_train'][:100]
    # y = D['y_train'][:100]
    # P = D
    # margs = {'voice_num': voice_num}
    # from chorales.sample import write_sample
    # write_sample(X, y, P, 'tmp2', '../data/output', True, margs, postfix='')
    return D

if __name__ == '__main__':
    # standardize_key('../data/input/JSB Chorales_parts.pickle', '../data/input/JSB Chorales_parts_Cs1.pickle')
    load('../data/input/JSB Chorales_parts_with_holds.pickle', seq_length=2, use_beats=True)
