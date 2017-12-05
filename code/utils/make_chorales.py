import numpy as np
import cPickle
from music21 import corpus

def pianoroll_to_song(roll, offset=21):
    f = lambda x: (np.where(x)[0]+offset).tolist()
    return [f(s) for s in roll]

def chorale_to_pianoroll(chorale, mult, nt=None, get_beats=False, hold_notes=False):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.

	could also have used graph.PlotHorizontalBarPitchSpaceOffset(chorale)

	if hold_notes, keep a binary vector specifying whether the note is being held or initiated
		- maybe assume only one note played at a time
	if get_beats, also tracks beat position of each note
	"""
	xss = chorale.chordify()
	if nt is None:
		nt = int(xss.duration.quarterLength*mult)+1
	nd = 128
	if hold_notes:
		nd += 1 # add extra note number
	roll = np.zeros((nt, nd))
	beats = np.nan*np.ones(nt)
	for xs in xss.flat.notes: # e.g. xs is <music21.chord.Chord G4 D4 B3 G2>
		# onset of note, in time steps
		ct = int(xs.offset*mult)

		# keep track of played notes
		ps = [n.midi for n in xs.pitches]
		roll[ct,ps] = 1

		# keep track of held notes
		if hold_notes:
			dur = int(xs.duration.quarterLength*mult)
			for i in range(1,dur):
				roll[ct+i,-1] = 1
				# is_held[ct+i] = True

		# quantize beat, e.g., 3.5 -> 3.0 if mult==1
		# beats[ct] = int(xs.beat*mult)*1.0/float(mult)
		beats[ct] = xs.beat

	# return desired number of outputs
	if get_beats:
		return roll, beats
	else:
		return roll

def make_beats(nbeats, start_beat, maxbeat, mult):
	delta = 1/float(mult)
	beats = ((np.cumsum(delta*np.ones(nbeats)) - delta) + (start_beat-1)) % maxbeat + 1
	return beats

def make_beats_from_template(all_beats, mult):
	B = np.vstack(all_beats).T
	nbeats = len(B)
	
	# find first non-nan beat count
	ind = 0
	while np.isnan(np.vstack(all_beats).T[ind]).all():
		ind += 1
	start_beat = np.nanmax(B[ind])
	start_beat -= ind/float(mult) # correct for incremented ind

	maxbeat = np.nanmax(B).astype(int)
	return make_beats(nbeats, start_beat, maxbeat, mult)

def parse_chorales_with_parts(mult, n=None):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.
	"""
	i = 0
	songs = []
	keys = []
	timesigs = []
	Beats = []
	for chorale in corpus.chorales.Iterator():
		if n is not None and i >= n:
			break
		print i+1
		key = chorale.analyze('key')
		timesig = chorale.flat.timeSignature.beatCount

		T = int(chorale.duration.quarterLength*mult)+1
		cursongs = []
		maxT = 0
		names = ['soprano', 'alto', 'tenor', 'bass']
		all_beats = []
		for part in chorale.parts:
			ind = [x for x in names if x in str(part).lower()]
			if not ind and i not in [5, 13, 149, 164, 166]:
				# manually confirmed these chorales are fine
				continue
			roll, beats = chorale_to_pianoroll(part, mult=mult, nt=T, hold_notes=True, get_beats=True)
			all_beats.append(beats)

			# keep track of actual length
			curT = np.where(roll.sum(axis=-1) > 0)[0].max()+1
			maxT = max(maxT, curT)

			# convert to song
			song = pianoroll_to_song(roll, offset=0)
			cursongs.append(song)

		# expecting four parts; skip otherwise
		if len(cursongs) != 4:
			print i, [str(p) for p in chorale.parts]
			continue
		
		# join beats and holds
		beats = make_beats_from_template(all_beats, mult)

		# shorten if T was too long
		beats = beats[:maxT]
		cursongs = [song[:maxT] for song in cursongs]

		# combine parts (-1 means silent)
		cursongs = [[y[0] if y else -1 for y in s] for s in zip(*cursongs)]
		songs.append(cursongs)
		Beats.append(beats)
		keys.append(key)
		timesigs.append(timesig)
		i += 1
	return {'songs': songs, 'keys': keys, 'beats': Beats,
		'timesigs': timesigs}

def parse_chorales(mult):
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
	return {'songs': songs, 'keys': keys}

def get_keys_as_modes(keys):
	return [k.mode == 'major' for k in keys]

def get_keys_as_names(keys):
	return [k.tonicPitchNameWithCase for k in keys]

def update_keys_with_modes(out):
	out['modes'] = get_keys_as_modes(out['keys'])
	out['keys'] = get_keys_as_names(out['keys'])
	return out

def cv_data_split(data, (tr,va,te)):

	# make train/test/valid inds, and shuffle data
	N = len(data['songs'])
	tr_ind = int(np.ceil(N*tr))
	te_ind = tr_ind + int(np.ceil(N*te))
	inds = np.arange(N)
	np.random.shuffle(inds)

	# subset all data into train/test/valid
	D = {}
	for k in data:
		items = np.array(data[k])[inds].tolist()
		postfix = '_' + k if k != 'songs' else ''
		D['train' + postfix] = items[:tr_ind]
		D['test' + postfix] = items[tr_ind:te_ind]
		D['valid' + postfix] = items[te_ind:]
	return D

def main(outfile='../data/input/JSB Chorales_parts_with_holds', keep_parts=True, mult=2):
	"""
	mult == 1 means quarter notes, mult == 2 means eighth notes, etc.
	"""
	if keep_parts:
		out = parse_chorales_with_parts(mult=mult)
	else:
		out = parse_chorales(mult=mult)
	out = update_keys_with_modes(out)
	D = cv_data_split(out, (0.6, 0.2, 0.2))
	cPickle.dump(D, open(outfile + '.pickle', 'wb'))

if __name__ == '__main__':
	main()
