import soundfile as sf
import tensorflow as tf
import numpy as np
from os import listdir, getcwd, rename, remove
from os.path import isfile, join
    
class Song(tf.Module):
    def __init__(self, sample_bank, length_in_beats, bpm, sr):
        self.soundbank_ref = sample_bank.soundbank_ref
        self.instruments = sample_bank.instruments
        self.num_inst_samples = len(sample_bank.soundbank)
        self.length_in_beats = length_in_beats
        self.bpm = bpm
        self.sr = sr
        N = self.sr * self.length_in_beats * 60 // self.bpm
        samples_per_beat = (self.sr * 60) // self.bpm
        padded_samples = []
        for i in range(len(sample_bank.soundbank)):
            sample = sf.read(sample_bank.soundbank[i])[0][:N,0]
            padded_sample = np.zeros(samples_per_beat * self.length_in_beats)
            padded_sample[:sample.shape[0]] = sample
            padded_samples.append(padded_sample)
        self.padded_samples = tf.constant(padded_samples, dtype=tf.float32)
        self.output = tf.Variable(tf.zeros([N,], dtype=tf.float32))
        self.notes = tf.Variable(tf.zeros([self.length_in_beats, self.num_inst_samples], dtype=tf.int32))
        self.notes_float = tf.Variable(tf.zeros([self.length_in_beats, self.num_inst_samples,], dtype=tf.float32))
        self.zero()

    @tf.function
    def zero(self):
        self.notes = self.notes.assign(tf.zeros([self.length_in_beats, self.num_inst_samples], dtype=tf.int32))
        self.notes_float = self.notes_float.assign(tf.zeros([self.length_in_beats, self.num_inst_samples,], dtype=tf.float32))
        samples_per_beat = (self.sr * 60) // self.bpm
        N = samples_per_beat * self.length_in_beats
        self.output.assign(tf.zeros([N,], dtype=tf.float32))
    
    #@tf.function
    def create(self):
        length = 3 * self.length_in_beats // 4
        N = self.sr * self.length_in_beats * 60 // self.bpm
        samples_per_beat = (self.sr * 60) // self.bpm
        self.notes = tf.random.uniform([length, self.num_inst_samples], minval=0, maxval=40, dtype=tf.int32)
        self.notes_float.assign(tf.zeros([self.length_in_beats, self.num_inst_samples], dtype=tf.float32))
        self.output.assign(tf.zeros([N,], dtype=tf.float32))
        for i in range(length):
            for j in range(self.num_inst_samples):
                if self.notes[i,j] == 0:
                    self.output.assign(tf.roll(self.padded_samples[j], shift=i*samples_per_beat, axis=0))
                    self.notes_float[i,j].assign(1.0)
        return self.output, self.notes_float
    
    def generate(self, instruments=None):
        if instruments is None:
            instruments = self.instruments
        length = 3 * self.length_in_beats // 4
        lst = [k for k in range(self.num_inst_samples) if self.soundbank_ref[k][0] in instruments]
        for i in range(length):
            make_note = np.random.choice([0 for i in range(9)] + [1])
            if make_note:
                j = np.random.choice(lst)
                intensity = 127 #np.random.randint(1,127)
                self.add(i, j, intensity)
        
    @tf.function                
    def __call__(self, notes_float, padded_samples):
        samples_per_beat = (self.sr * 60) // self.bpm
        N = samples_per_beat * self.length_in_beats
        self.zero()
        for i in range(self.length_in_beats):
            for j in range(self.num_inst_samples):
                self.output.assign_add(notes_float[i,j] * tf.roll(padded_samples[j], shift=i*samples_per_beat, axis=0))
        return None
    
    #@tf.function
    def add(self, i, j, intensity):
        samples_per_beat = (self.sr * 60) // self.bpm
        old_intensity = self.notes[i,j]
        new_intensity = tf.cast(intensity, dtype=tf.int32)
        float_intensity = tf.cast(intensity, dtype=tf.float32) / 128.0
        float_intensity_difference = tf.cast((new_intensity - old_intensity), dtype=tf.float32) / 128.0
        self.output = self.output.assign_add(float_intensity_difference * tf.roll(self.padded_samples[j], shift=i*samples_per_beat, axis=0))
        self.notes_float = self.notes_float[i,j].assign(float_intensity)
        self.notes = self.notes[i,j].assign(new_intensity)
        
    def transform(self, matrix):
        for i in range(self.notes.shape[0]):
            self.notes[i] = tf.matmul(matrix, self.notes[i])

class SampleBank(tf.Module):
    def __init__(self, restricted_instruments = None, name=None):
        super(SampleBank, self).__init__(name=name)
        path = join(getcwd(), 'preprocessed_samples')
        files = [f for f in listdir(path) if isfile(join(path, f)) and f[0] != '.']
        samples = []
        for f in files:
            sample = {'instrument' : '_'.join(f.split('_')[:-1]),
                      'midi_number' : int(f.split('_')[-1][:-4]),
                      'filename' : join(path, f)}
            samples.append(sample)
        samples = sorted(samples, key=lambda x: (x['instrument'], x['midi_number']))
    
        instrument_names = set([s['instrument'] for s in samples])
        instruments = {}
        for inst in instrument_names:
            samples_for_inst = {s['midi_number'] : s['filename'] for s in samples if s['instrument'] == inst}
            instruments[inst] = {'samples' : samples_for_inst,
                                 'min_note' : min(samples_for_inst.keys()),
                                 'max_note' : max(samples_for_inst.keys())}
        if not restricted_instruments:
            restricted_instruments = [
                inst for inst in instrument_names
                if len(instruments[inst]['samples']) == 1 + (instruments[inst]['max_note'] - instruments[inst]['min_note'])
            ]
        soundfont = {}
        soundbank_ref = []
        k = 0
        for inst in restricted_instruments:
            print(inst, ' --- notes ',instruments[inst]['min_note'], ' through ', instruments[inst]['max_note'])
            soundfont[inst] = {}
            for i in range(128):
                if i in instruments[inst]['samples'].keys():
                    soundfont[inst][i] = (k, instruments[inst]['samples'][i])
                    soundbank_ref.append((inst, i))
                    k += 1
                else:
                    soundfont[inst][i] = None
        soundbank = [soundfont[x[0]][x[1]][1] for x in soundbank_ref]
        self.soundbank = soundbank
        self.soundbank_ref = soundbank_ref
        self.instruments = restricted_instruments