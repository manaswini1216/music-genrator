import streamlit as st
from midi import MIDI
from model import MODEL
import numpy as np
import os

st.title("Music Generation with C-RNN-GAN")

midi_obj = MIDI(seq_length=100)
model = MODEL(midi_obj)

if os.path.exists("Model/generator.h5"):
    model.generator.load_weights("Model/generator.h5")
    st.success("Model loaded successfully!")
else:
    st.warning("Model weights not found! Please train the model or upload weights.")

if st.button("Generate Music"):
    st.info("Generating music...")
    model.generate()  
    st.success("Music generated!")

    st.write("Generated MIDI file: Result/gan_final.mid")
