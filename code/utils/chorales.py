import numpy as np
import cPickle
from music21 import corpus
from utils.pianoroll import pianoroll_to_song

def chorale_to_pianoroll(chorale, mult=1, nt=None):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.

	could also have used graph.PlotHorizontalBarPitchSpaceOffset(chorale)
	"""
	xss = chorale.chordify()
	if nt is None:
		nt = int(xss.duration.quarterLength*mult)+1
	roll = np.zeros((nt, 128))
	for xs in xss.flat.notes: # e.g. xs is <music21.chord.Chord G4 D4 B3 G2>
		ct = int(xs.offset*mult)
		ps = [n.midi for n in xs.pitches] # e.g. n is <music21.pitch.Pitch G4>
		roll[ct,ps] = 1
	if roll[-1].sum() == 0: # remove last frame if empty
		roll = roll[:-1]
	if roll[0].sum() == 0: # remove first frame if empty
		roll = roll[1:]
	return roll

def parse_chorales_with_parts(mult=1):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.
	"""
	i = 0
	songs = []
	keys = []
	for chorale in corpus.chorales.Iterator():
		key = chorale.analyze('key')
		T = int(chorale.duration.quarterLength*mult)+1
		cursongs = []
		maxT = 0
		names = ['soprano', 'alto', 'tenor', 'bass']
		for part in chorale.parts:
			ind = [x for x in names if x in str(part).lower()]
			if not ind and i not in [5, 13, 149, 164, 166]:
				# manually confirmed these chorales are fine
				continue
			roll = chorale_to_pianoroll(part, mult=1, nt=T)

			# keep track of actual length
			curT = np.where(roll.sum(axis=-1) > 0)[0].max()+1
			maxT = max(maxT, curT)

			# convert to song
			song = pianoroll_to_song(roll, offset=0)
			cursongs.append(song)
		if len(cursongs) != 4:
			print i, [str(p) for p in chorale.parts]
			continue
		# shorten if T was too long
		cursongs = [song[:maxT] for song in cursongs]

		# combine parts (-1 means silent)
		cursongs = [[y[0] if y else -1 for y in s] for s in zip(*cursongs)]
		songs.append(cursongs)
		keys.append(key)
		i += 1
	return songs, keys

def parse_chorales(mult=1):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.
	"""
	i = 0
	songs = []
	keys = []
	for chorale in corpus.chorales.Iterator():
		key = chorale.analyze('key')
		roll = chorale_to_pianoroll(chorale, mult)
		song = pianoroll_to_song(roll, offset=0)
		songs.append(song)
		keys.append(key)
		i += 1
	return songs, keys

def add_keys(D):
	E = {}
	kys = []
	for ky in D:
		if '_keys' in ky:
			continue
		E[ky] = D[ky]
		E[ky + '_mode'] = [k.mode == 'major' for k in D[ky + '_keys']]
		E[ky + '_key'] = [k.tonicPitchNameWithCase for k in D[ky + '_keys']]
	return E

def cv_data_split(X, keys, (tr,va,te)):
	N = len(X)
	tr_ind = int(np.ceil(N*tr))
	te_ind = tr_ind + int(np.ceil(N*te))
	inds = np.arange(N)
	np.random.shuffle(inds)
	X = np.array(X)[inds].tolist()
	keys = np.array(keys)[inds].tolist()
	D = {}
	D['train'] = X[:tr_ind]
	D['train_keys'] = keys[:tr_ind]
	D['test'] = X[tr_ind:te_ind]
	D['test_keys'] = keys[tr_ind:te_ind]
	D['valid'] = X[te_ind:]
	D['valid_keys'] = keys[te_ind:]
	return add_keys(D)

def main(outfile='../data/input/JSB Chorales_parts2', keep_parts=True, mult=2):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.
	"""
	if keep_parts:
		songs, keys = parse_chorales_with_parts(mult=mult)
	else:
		songs, keys = parse_chorales(mult=mult)
	D = cv_data_split(songs, keys, (0.6, 0.2, 0.2))
	cPickle.dump(D, open(outfile + '.pickle', 'wb'))
	# np.savez(outfile, songs=songs, keys=keys)

if __name__ == '__main__':
	main()
