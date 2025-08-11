import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from music21 import *
from keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils


class MIDI():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.file_notes = []
        self.trainseq = []
        self.transfer_dic = dict()
        self.dic_n = 0

    def parser(self, folderName):
        """Parse all MIDI files in the folder and extract notes, rests, and chords."""
        self.file_notes = []  
        for file in glob.glob(f"{folderName}/*.mid"):
            midi = converter.parse(file)
            print(f"Parsing {file}")

            notes = []
            for element in midi.flat.elements:
                if isinstance(element, note.Rest) and element.offset != 0:
                    notes.append('R')
                elif isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(pitch) for pitch in element.pitches))

            self.file_notes.append(notes)

        note_set = sorted(set(note for notes in self.file_notes for note in notes))
        self.dic_n = len(note_set)

        self.transfer_dic = {note: number for number, note in enumerate(note_set)}

    def prepare_sequences(self):
        """Create input sequences normalized between -1 and 1 for training."""
        self.trainseq = []  
        for notes in self.file_notes:
            for i in range(len(notes) - self.seq_length):
                seq_in = [self.transfer_dic[note] for note in notes[i:i + self.seq_length]]
                self.trainseq.append(seq_in)

        self.trainseq = np.array(self.trainseq)
        self.trainseq = (self.trainseq - float(self.dic_n) / 2) / (float(self.dic_n) / 2)

        return self.trainseq

    def create_midi(self, prediction_output, filename):
        """Convert predicted output to MIDI file."""
        offset = 0
        midi_stream = stream.Stream()

        for pattern in prediction_output:
            if pattern == 'R':
                midi_stream.append(note.Rest(quarterLength=0.5))
            elif '.' in pattern or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                midi_stream.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                midi_stream.append(new_note)

            offset += 0.5

        midi_stream.write('midi', fp=f'{filename}.mid')
