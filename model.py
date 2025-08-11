from MIDI import MIDI
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, LSTM, Bidirectional, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

class MODEL():
    def __init__(self, midi_obj):
        self.midi = midi_obj
        self.seq_length = self.midi.seq_length
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss = []

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim, 1))
        generated_seq = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(generated_seq)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)

    def build_generator(self):
        model = Sequential()
        model.add(LSTM(512, input_shape=(self.latent_dim, 1), return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))

        noise = Input(shape=(self.latent_dim, 1))
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, dataFolder, batch_size=128, sample_interval=50):
        self.midi.parser(dataFolder)
        sequences = self.midi.prepare_sequences()

        print(f"\nNumber of sequences for training: {sequences.shape[0]}\n")

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, sequences.shape[0], batch_size)
            real_seqs = sequences[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)

            if epoch % sample_interval == 0:
                print(f"{epoch + 1} / {epochs} [D loss: {d_loss[0]:.6f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.6f}]")
                self.disc_loss.append(d_loss[0])
                self.gen_loss.append(g_loss)

        print(f"The C-RNN-GAN model has been trained with {dataFolder} MIDI music.\n" +
              "You can save the model using the save() method.")

    def save(self):
        if not os.path.exists('Model/'):
            os.makedirs('Model/')
        self.discriminator.save('Model/discriminator.h5')
        self.generator.save('Model/generator.h5')
        print("Saved discriminator and generator models in 'Model/' folder.")

    def generate(self):
        """Generate music using the trained generator."""
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)

        boundary = int(len(self.midi.transfer_dic) / 2)
        pred_nums = [x * boundary + boundary for x in predictions[0]]
        notes = list(self.midi.transfer_dic.keys())
        pred_notes = [notes[int(x)] for x in pred_nums]

        if not os.path.exists('Result/'):
            os.makedirs('Result/')

        self.midi.create_midi(pred_notes, 'Result/gan_final')

    def plot_loss(self):
        plt.plot(self.disc_loss, color='red')
        plt.plot(self.gen_loss, color='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('Result/GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.show()
        plt.close()
