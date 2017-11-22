import pickle
import numpy as np
from music21 import key
from keras.utils import to_categorical
from utils.midi_utils import write_song

def findOffsetToStandardize(curKey, keyGoal):
    if type(curKey) is str:
        curKey = key.Key(curKey)
    i = intervalToKey(curKey, keyGoal)
    return i.chromatic.semitones, curKey.mode == key.Key(keyGoal).mode

def change_key_of_song(song, curKey, goalKey, const=0):
    offset, modeMatch = findOffsetToStandardize(curKey, goalKey)
    return [[y+offset+const if y!=-1 else y for y in x] for x in song]

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
            X2[X2 == -1] = 1e5
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
        X2[X2 == -1] = 1e4
        mns.append(X2.min(axis=0))
        mxs.append(X.max(axis=0))
    mns = np.vstack(mns).min(axis=0)
    mxs = np.vstack(mxs).max(axis=0)
    return mns, mxs-mns+1

def make_hist_and_offset(d, seq_length, batch_size):
    """
    build history terms and subtract min note number
    """
    offsets, ranges = find_note_ranges(d)
    D = {'offsets': offsets, 'ranges': ranges}
    for k in ['train', 'test', 'valid']:
        # add history terms
        xss = []
        for song in d[k]:
            # -2 means not given, or nan
            xs = np.dstack([np.roll(song, i, axis=0) for i in xrange(seq_length)])
            xs[:(seq_length-1),:,1:] = -2 # dne
            xss.append(xs)
        X = np.vstack(xss)
        X = X.swapaxes(1,2) # [n x seq_length x 4]
        
        # shift so everything starts at 0
        ix1 = X == -1
        ix2 = X == -2
        X -= (offsets-2) # now all start at 2
        X[ix1] = 1 # -1 -> 1 for silence
        X[ix2] = 0 # -2 -> 0 for dne

        # ignore ends to be divisible by batch size
        n = (len(X) / batch_size)*batch_size
        X = X[:n]

        D[k] = X
    return D

def X_to_onehot(X, num_notes):
    xs = []
    for i in xrange(X.shape[-1]):
        xcs = []
        for j in xrange(X.shape[1]):
            xc = to_categorical(X[:,j,i], num_classes=num_notes[i]+2)
            xcs.append(xc)
        xc = np.dstack(xcs).swapaxes(1,2)
        xc = xc[:,:,1:] # remove dne col -> all zeros
        xs.append(xc)
    return np.dstack(xs) # [n x seq_length x nnotes]

def y_to_onehot(y, num_notes):
    y = to_categorical(y, num_classes=num_notes+2)
    return y[:,1:] # remove dne col -> all zeros

def split_X_and_y(D, yind, inds_to_zero=None, make_onehot=True):
    """
    separate X from y
    """
    for k in ['train', 'test', 'valid']:
        X = D[k]
        # make X and Y
        y = X[:,0,yind].copy()
        if inds_to_zero is not None:
            X[:,0,inds_to_zero] = 0 # 0 -> dne
        X[:,0,yind] = 0 # 0 -> dne
        if make_onehot:
            y = y_to_onehot(y, D['ranges'][yind])
            X = X_to_onehot(X, D['ranges'])
        D['x_' + k] = X
        D['y_' + k] = y
    return D

def onehot_to_y(y, offset):
    row, ind = np.where(y)
    if len(row) == 0:
        # zero'd out part
        return -1*np.ones(len(y))
    assert len(ind) == len(y)
    ix0 = ind == 0
    ind += (offset-1)
    ind[ix0] = -1
    return ind

def X_and_y_to_song(X, y, yind, offsets, ranges):
    x = X[:,0,:] # drop history
    voice_inds = np.hstack([i*np.ones(r+1) for i,r in enumerate(ranges)])
    song = []
    for i,offset in enumerate(offsets):
        # each voice is a matrix of one-hot vectors
        if i == yind:
            part = y
        else:
            part = x[:,voice_inds == i]
        # convert one-hot vectors to note indices
        inds = onehot_to_y(part, offset)
        song.append(inds)
    return np.vstack(song).T # [n x 4]

def test_songs_are_invertible(D, d):
    song0 = np.vstack([np.vstack(x) for x in d['train']])
    song1 = X_and_y_to_song(D['x_train'], D['y_train'], voice_num, D['offsets'], D['ranges'])
    assert (song0 - song1).sum() == 0

def write_songs(songs):
    for i, song in enumerate(songs):
        fnm = '../data/output/sample_{}.mid'.format(i)
        write_song(song, fnm, isHalfAsSlow=True)

def load(train_file='../data/input/JSB Chorales_parts.pickle', voice_num=0, seq_length=8, batch_size=1, voices_to_zero=[1]):
    """
    load data by parts
    where you predict one part using the other parts
        plus the sequence history
    """
    d = pickle.load(open(train_file))
    D = make_hist_and_offset(d, seq_length, batch_size)
    D = split_X_and_y(D, voice_num, inds_to_zero=voices_to_zero)
    write_songs(d['train'][:10])
    return D

if __name__ == '__main__':
    # standardize_key('../data/input/JSB Chorales_parts.pickle', '../data/input/JSB Chorales_parts_Cs1.pickle')
    load()
