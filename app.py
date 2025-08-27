import streamlit as st
import numpy as np
import os
import pickle
from keras.models import load_model
from music21 import instrument, stream, note, chord

st.set_page_config(page_title="Music GAN Generator", layout="centered")
st.title("🎹 Music GAN Generator — Generate & Download MIDI")

MODEL_PATH = "gan_final.h5"
TRANSFER_PKL = "transfer_dic.pkl"
OUTPUT_FOLDER = "static"
OUTPUT_FILENAME = "generated_music.mid"
BASE_QUARTER = 0.5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def make_fallback_transfer():
    octaves = [3, 4, 5]
    notes = []
    chroma = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for o in octaves:
        for p in chroma:
            notes.append(f"{p}{o}")
    notes.append('R')
    return {n: i for i, n in enumerate(notes)}

def load_transfer(transfer_path=TRANSFER_PKL):
    if os.path.exists(transfer_path):
        try:
            with open(transfer_path, "rb") as f:
                d = pickle.load(f)
            st.success(f"Loaded transfer mapping from {transfer_path} (size={len(d)})")
            return d
        except Exception as e:
            st.warning(f"Failed to load {transfer_path}: {e}. Using fallback mapping.")
            return make_fallback_transfer()
    else:
        st.info(f"No {transfer_path} found in repo — using fallback mapping.")
        return make_fallback_transfer()

def create_midi_from_indices(indices, transfer_dic, out_path):
    number_to_note = {v:k for k,v in transfer_dic.items()}
    if not out_path.lower().endswith(".mid"):
        out_path += ".mid"

    midi_stream = stream.Stream()
    offset = 0.0
    for idx in indices:
        try:
            idx = int(idx)
        except:
            continue
        pattern = number_to_note.get(idx, 'C4')

        # small probability for rest
        if np.random.rand() < 0.05:
            pattern = 'R'

        # mostly fixed duration
        dur = 0.5
        if np.random.rand() < 0.2:
            dur = np.random.choice([0.25, 0.75])

        if pattern == 'R':
            r = note.Rest(quarterLength=dur)
            r.offset = offset
            midi_stream.append(r)
        elif '.' in pattern:
            parts = pattern.split('.')
            notes_list = []
            for p in parts:
                try:
                    n = note.Note(p)
                    n.storedInstrument = instrument.Piano()
                    n.quarterLength = dur
                    # occasional octave shift
                    if np.random.rand() < 0.1:
                        n.octave += np.random.choice([-1,1])
                    notes_list.append(n)
                except:
                    pass
            if notes_list:
                ch = chord.Chord(notes_list)
                ch.offset = offset
                ch.quarterLength = dur
                midi_stream.append(ch)
        else:
            try:
                n = note.Note(pattern)
                n.offset = offset
                n.storedInstrument = instrument.Piano()
                n.quarterLength = dur
                if np.random.rand() < 0.1:
                    n.octave += np.random.choice([-1,1])
                midi_stream.append(n)
            except:
                r = note.Rest(quarterLength=dur)
                r.offset = offset
                midi_stream.append(r)

        offset += dur

    midi_stream.write('midi', fp=out_path)
    return out_path


def make_noise_for_model(model):
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    dims = tuple(d for d in ishape[1:] if d is not None)
    if len(dims) == 0:
        dims = (100,)
    noise_shape = (1,) + dims
    return np.random.normal(0,1,size=noise_shape).astype(np.float32), noise_shape

def prediction_to_indices(pred_array, n_tokens):
    # Add tiny noise for variety
    pred_array = pred_array + np.random.normal(0, 0.05, size=pred_array.shape)
    seq = pred_array
    smin, smax = float(seq.min()), float(seq.max())
    if smin >= -1.0 - 1e-6 and smax <= 1.0 + 1e-6:
        idxs = np.clip(np.round(((seq + 1.0)/2.0) * (n_tokens - 1)).astype(int), 0, n_tokens - 1)
    else:
        if smax == smin:
            idxs = np.random.randint(0, n_tokens, size=max(8, seq.size))
        else:
            norm = (seq - smin) / (smax - smin)
            idxs = np.clip(np.round(norm * (n_tokens - 1)).astype(int), 0, n_tokens - 1)
    return idxs

transfer_dic = load_transfer()

@st.cache_resource
def load_generator_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Upload gan_final.h5 to the repo root.")
    model = load_model(path, compile=False)
    return model

try:
    model = load_generator_model()
    st.success("Generator model loaded.")
except Exception as e:
    st.error(f"Could not load generator model: {e}")
    st.stop()

st.write("Model input shape:", model.input_shape)
st.write("Transfer mapping size (tokens):", len(transfer_dic))

st.markdown("Press **Generate** to create a MIDI and then use **Download** to save it.")
num_notes = st.slider("Select number of notes", min_value=50, max_value=600, value=300, step=50)

if st.button("Generate"):
    noise, noise_shape = make_noise_for_model(model)
    chunk_size = noise_shape[1]
    total_chunks = int(np.ceil(num_notes / chunk_size))

    predictions = []
    for _ in range(total_chunks):
        try:
            pred = model.predict(noise, verbose=0)
        except:
            pred = model(noise, training=False).numpy()

        if isinstance(pred, np.ndarray):
            if pred.ndim == 3:
                seq = pred[0].squeeze()
                if seq.ndim == 2:
                    seq = seq.mean(axis=-1)
            elif pred.ndim == 2:
                seq = pred[0]
            else:
                seq = pred.flatten()
        else:
            st.error("Unexpected prediction type.")
            st.stop()

        predictions.append(seq)

    full_seq = np.concatenate(predictions)[:num_notes]
    indices = prediction_to_indices(full_seq, len(transfer_dic))
    out_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    created = create_midi_from_indices(indices, transfer_dic, out_path)

    with open(created, "rb") as f:
        midi_bytes = f.read()
    st.success("✅ MIDI generated successfully!")
    st.download_button("⬇️ Download MIDI", data=midi_bytes, file_name="generated_music.mid", mime="audio/midi")
