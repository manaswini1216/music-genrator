# app.py
import streamlit as st
import numpy as np
import os
import pickle
from keras.models import load_model
from music21 import instrument, stream, note, chord
import tempfile

st.set_page_config(page_title="Music GAN Generator", layout="centered")
st.title("🎹 Music GAN Generator — Generate & Download MIDI")

# ---------- CONFIG ----------
MODEL_PATH = "gan_final.h5"
TRANSFER_PKL = "transfer_dic.pkl"   # if available from training, place in repo root
OUTPUT_FOLDER = "static"
OUTPUT_FILENAME = "generated_music.mid"
DURATION_QUARTER = 0.5  # quarterLength per token

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- Helpers ----------
def make_fallback_transfer():
    """Create a fallback mapping of note->index (chromatic across octaves).
       This ensures generation produces non-empty MIDI even if training mapping is missing.
       Not a substitute for the real transfer_dic used during training."""
    octaves = [3, 4, 5]  # C3..B5
    notes = []
    chroma = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for o in octaves:
        for p in chroma:
            notes.append(f"{p}{o}")
    # add a rest token
    notes.append('R')
    transfer = {n:i for i,n in enumerate(notes)}
    return transfer

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
        st.info(f"No {transfer_path} found in repo — using fallback mapping (not identical to training).")
        return make_fallback_transfer()

def create_midi_from_indices(indices, transfer_dic, out_path):
    """Build a music21 Stream from integer indices and write to out_path (.mid)."""
    # reverse mapping
    number_to_note = {v:k for k,v in transfer_dic.items()}
    # ensure extension
    if not out_path.lower().endswith(".mid"):
        out_path = out_path + ".mid"

    midi_stream = stream.Stream()
    offset = 0.0
    for idx in indices:
        # safe convert
        try:
            idx = int(idx)
        except Exception:
            continue
        pattern = number_to_note.get(idx, 'C4')
        # set explicit duration
        if pattern == 'R':
            r = note.Rest(quarterLength=DURATION_QUARTER)
            r.offset = offset
            midi_stream.append(r)
        elif isinstance(pattern, str) and '.' in pattern:
            parts = pattern.split('.')
            notes = []
            for p in parts:
                try:
                    n = note.Note(p)
                    n.storedInstrument = instrument.Piano()
                    n.quarterLength = DURATION_QUARTER
                    notes.append(n)
                except:
                    pass
            if notes:
                ch = chord.Chord(notes)
                ch.offset = offset
                ch.quarterLength = DURATION_QUARTER
                midi_stream.append(ch)
        else:
            try:
                n = note.Note(pattern)
                n.offset = offset
                n.storedInstrument = instrument.Piano()
                n.quarterLength = DURATION_QUARTER
                midi_stream.append(n)
            except:
                # fallback rest if invalid
                r = note.Rest(quarterLength=DURATION_QUARTER)
                r.offset = offset
                midi_stream.append(r)
        offset += DURATION_QUARTER

    midi_stream.write('midi', fp=out_path)
    return out_path

def make_noise_for_model(model):
    """Create gaussian noise with shape matching model.input_shape (batch=1)."""
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    # drop batch dim
    dims = tuple(d for d in ishape[1:] if d is not None)
    if len(dims) == 0:
        dims = (100,)  # fallback
    noise_shape = (1,) + dims
    return np.random.normal(0,1,size=noise_shape).astype(np.float32), noise_shape

def prediction_to_indices(pred_array, n_tokens):
    """Convert model output (floats) to integer token indices.
       Model was trained with normalization: (idx - dic_n/2) / (dic_n/2) -> roughly [-1,1].
       So invert using tanh assumption first; otherwise normalize generically."""
    seq = pred_array
    # If pred in [-1,1] typical tanh:
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

# ---------- Load transfer mapping ----------
transfer_dic = load_transfer()

# ---------- Load model ----------
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

# ---------- UI ----------
st.markdown("Press **Generate** to create a MIDI and then use **Download** to save it.")

if st.button("Generate"):
    # create noise dynamically to match model input
    noise, noise_shape = make_noise_for_model(model)
    st.write("Noise shape used for generation:", noise_shape)

    # run inference
    try:
        pred = model.predict(noise, verbose=0)
    except Exception:
        # try calling directly
        try:
            pred = model(noise, training=False).numpy()
        except Exception as e:
            st.error("Model inference failed: " + str(e))
            st.stop()

    st.write("Raw prediction shape:", getattr(pred, "shape", str(type(pred))))

    # flatten to 1D time series depending on output dims
    if isinstance(pred, np.ndarray):
        if pred.ndim == 3:
            seq = pred[0].squeeze()
            if seq.ndim == 2:
                # average features (if any)
                seq = seq.mean(axis=-1)
        elif pred.ndim == 2:
            seq = pred[0]
        elif pred.ndim == 1:
            seq = pred
        else:
            seq = pred.flatten()
    else:
        st.error("Unexpected prediction type.")
        st.stop()

    st.write("Sequence length:", seq.shape)
    st.write("Prediction min/max:", float(seq.min()), float(seq.max()))

    # convert predictions to integer indices
    indices = prediction_to_indices(seq, len(transfer_dic))
    st.write("First 40 mapped indices:", indices[:40].tolist())

    # create midi in static folder
    out_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    try:
        created = create_midi_from_indices(indices, transfer_dic, out_path)
    except Exception as e:
        st.error("MIDI creation failed: " + str(e))
        st.stop()

    # check file size
    try:
        size = os.path.getsize(created)
    except Exception as e:
        st.error("Could not stat generated file: " + str(e))
        st.stop()

    st.write("Generated MIDI path:", created)
    st.write("MIDI file size (bytes):", size)

    if size == 0:
        st.error("Generated MIDI file is empty (0 bytes). This usually means the model output mapped to no valid tokens or music21 failed to write. If you have the original transfer_dic.pkl from training, add it to the repo and retry.")
    else:
        with open(created, "rb") as f:
            midi_bytes = f.read()
        st.success("MIDI generated — ready to download.")
        st.download_button("Download MIDI", data=midi_bytes, file_name="generated_music.mid", mime="audio/midi")

st.markdown("---")
st.markdown("⚠️ If the music sounds wrong or is still very short, it's because the app is using a fallback mapping. For best results, include the exact `transfer_dic.pkl` used during training (place in the repo root).")

