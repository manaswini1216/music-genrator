import streamlit as st
import numpy as np
from music21 import instrument, stream, note, chord
from keras.models import load_model
from midi2audio import FluidSynth
import os

# --- MIDI creation function ---
def create_midi(prediction_output, filename, transfer_dic):
    offset = 0
    midi_stream = stream.Stream()
    number_to_note = {v: k for k, v in transfer_dic.items()}

    for pattern_num in prediction_output:
        pattern = number_to_note.get(int(pattern_num), 'C4')

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

# --- Load GAN model ---
@st.cache_resource(show_spinner=False)
def load_generator_model():
    return load_model('gan_final.h5')

model = load_generator_model()

# --- Static note mapping dictionary ---
transfer_dic = {'C4': 0, 'D4': 1, 'E4': 2, 'R': 3}

st.title("🎹 Music GAN Generator with MIDI to Audio Playback")

if st.button("Generate Music"):
    latent_dim = 1000
    noise = np.random.normal(0, 1, (1, latent_dim, 1))
    prediction = model.predict(noise)[0]

    boundary = int(len(transfer_dic) / 2)
    pred_nums = np.clip((prediction * boundary + boundary).astype(int), 0, len(transfer_dic) - 1)

    midi_filename = "generated_music.mid"
    wav_filename = "generated_music.wav"

    # Create MIDI file
    create_midi(pred_nums, midi_filename, transfer_dic)

    # Convert MIDI to WAV
    # Make sure FluidSynth can find your soundfont file (change path if needed)
    soundfont_path = 'FluidR3_GM.sf2'  # put your .sf2 in the same folder or provide full path
    if not os.path.exists(soundfont_path):
        st.error(f"Soundfont file '{soundfont_path}' not found! Please upload it in the app directory.")
    else:
        fs = FluidSynth(sound_font=soundfont_path)
        fs.midi_to_audio(midi_filename, wav_filename)

        # Play WAV audio in Streamlit
        with open(wav_filename, 'rb') as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/wav')

    # Provide MIDI file download
    with open(midi_filename, 'rb') as f:
        midi_bytes = f.read()

    st.download_button("Download MIDI file", midi_bytes, file_name=midi_filename, mime='audio/midi')

