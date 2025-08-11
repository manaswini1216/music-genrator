import streamlit as st
import numpy as np
from music21 import instrument, stream, note, chord
from keras.models import load_model
import os

def create_midi(prediction_output, filename, transfer_dic):
    offset = 0
    midi_stream = stream.Stream()

    number_to_note = {v: k for k, v in transfer_dic.items()}

    for pattern_num in prediction_output:
        pattern = number_to_note.get(pattern_num, 'C4')  

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

    midi_stream.write('midi', fp=filename)

transfer_dic = {'C4': 0, 'D4': 1, 'E4': 2, 'R': 3}

@st.cache(allow_output_mutation=True)
def load_generator_model():
    return load_model('gen_final.h5')

model = load_generator_model()

st.title("🎹 Music GAN Generator")
st.write("Generate new music using your trained GAN model.")

if st.button("Generate Music"):
    latent_dim = 1000
    noise = np.random.normal(0, 1, (1, latent_dim, 1))
    prediction = model.predict(noise)[0]

    boundary = int(len(transfer_dic) / 2)
    pred_nums = np.clip((prediction * boundary + boundary).astype(int), 0, len(transfer_dic)-1)

    midi_filename = "generated_music.mid"
    create_midi(pred_nums, midi_filename, transfer_dic)

    st.success("🎵 Music generated!")

    audio_file = open(midi_filename, 'rb').read()
    st.audio(audio_file, format='audio/midi')

    st.download_button(label="Download MIDI file", data=audio_file, file_name=midi_filename, mime='audio/midi')
