"""
source: https://github.com/yoavz/music_rnn
"""
import sys, os
import numpy as np
import midi

RANGE = 128

def song_to_pianoroll(song, offset=21):
    """
    song = [(60, 72, 79, 88), (72, 79, 88), ...]
    """
    rolls = []
    all_notes = [y for x in song for y in x if y != -1]
    if min(all_notes)-offset < 0:
        offset -= 12
        # assert False
    if max(all_notes)-offset > 87:
        offset += 12
        # assert False
    for notes in song:
        roll = np.zeros(88)
        roll[[int(n-offset) for n in notes if n != -1]] = 1.
        rolls.append(roll)
    return np.vstack(rolls)

class MidiWriter(object):

    def __init__(self, verbose=False, default_vel=100):
        self.verbose = verbose
        self.note_range = RANGE
        self.default_velocity = default_vel

    def note_off(self, val, tick):
        self.track.append(midi.NoteOffEvent(tick=tick, pitch=val))
        return 0

    def note_on(self, val, tick):
        self.track.append(midi.NoteOnEvent(tick=tick, pitch=val, velocity=self.default_velocity))
        return 0

    def dump_sequence_to_midi(self, seq, output_filename,
        time_step=120, resolution=480, metronome=24, offset=21,
        format='final', holds=None, is_song=True):
        if self.verbose:
            print "Dumping sequence to MIDI file: {}".format(output_filename)
            print "Resolution: {}".format(resolution)
            print "Time Step: {}".format(time_step)

        pattern = midi.Pattern(resolution=resolution)
        self.track = midi.Track()

        # metadata track
        meta_track = midi.Track()
        time_sig = midi.TimeSignatureEvent()
        time_sig.set_numerator(4)
        time_sig.set_denominator(4)
        time_sig.set_metronome(metronome)
        time_sig.set_thirtyseconds(8)
        meta_track.append(time_sig)
        pattern.append(meta_track)

        # reshape to (SEQ_LENGTH X NUM_DIMS)
        if format == 'icml':
            # assumes seq is list of lists, where each inner list are all the midi notes that were non-zero at that given timestep
            sequence = np.zeros([len(seq), self.note_range])
            sequence = [1 if i in tmstp else 0 for i in xrange(self.note_range) for tmstp in seq]
            sequence = np.reshape(sequence, [self.note_range,-1]).T
        elif format == 'flat':
            sequence = np.reshape(seq, [-1, self.note_range])
        else:
            sequence = seq

        time_steps = sequence.shape[0]
        if self.verbose:
            print "Total number of time steps: {}".format(time_steps)

        tick = time_step
        self.notes_on = { n: False for n in range(self.note_range) }
        for seq_idx in range(time_steps):
            if is_song:
                notes = sequence[seq_idx].astype(int)
                hold = holds[seq_idx] if holds is not None else []
                ix = notes >= 0
                notes = notes[ix]
                hold = hold[ix]
            else:
                notes = np.nonzero(sequence[seq_idx, :])[0].tolist()
            # n.b. notes += 21 ??
            # need to be in range 21,109
            notes = [n+offset for n in notes]

            # this tick will only be assigned to first NoteOn/NoteOff in
            # this time_step

            # NoteOffEvents come first so they'll have the tick value
            # go through all notes that are currently on and see if any
            # turned off
            for n in self.notes_on:
                # if note that was on last time but is now not here
                if self.notes_on[n] and n not in notes:
                    tick = self.note_off(n, tick)
                    self.notes_on[n] = False
                # turn off note if not being held
                if self.notes_on[n] and n in notes and holds is not None:
                    is_hold = [h for h,nt in zip(hold,notes) if nt==n][0]
                    if not is_hold:
                        tick = self.note_off(n, tick)
                        self.notes_on[n] = False

            # Turn on any notes that weren't previously on
            for note in notes:
                if not self.notes_on[note]:
                    tick = self.note_on(note, tick)
                    self.notes_on[note] = True

            tick += time_step

        # flush out notes
        for n in self.notes_on:
            if self.notes_on[n]:
                self.note_off(n, tick)
                tick = 0
                self.notes_on[n] = False

        pattern.append(self.track)
        midi.write_midifile(output_filename, pattern)

def write_sample(sample, fnm, isHalfAsSlow=False, holds=None):
    time_step = 240 if isHalfAsSlow else 120
    MidiWriter().dump_sequence_to_midi(sample, fnm,
        time_step=time_step, holds=holds)

def write_song(song, fnm, offset=21, isHalfAsSlow=False, holds=None):
    time_step = 240 if isHalfAsSlow else 120
    MidiWriter().dump_sequence_to_midi(song, fnm, offset=0,
        time_step=time_step, holds=holds, is_song=True)
