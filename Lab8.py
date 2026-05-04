import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Generate dummy dataset
data = np.random.rand(10000, 20)

input_dim = 20
latent_dim = 2

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(10, activation='relu')(inputs)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

decoder_h = Dense(10, activation='relu')
decoder_output = Dense(input_dim, activation='sigmoid')

h_decoded = decoder_h(z)
outputs = decoder_output(h_decoded)

vae = Model(inputs, outputs)

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        original_inputs, reconstructions, z_mean, z_log_var = inputs

        # Reconstruction loss (mean squared error)
        reconstruction_loss = tf.reduce_mean(
            tf.square(original_inputs - reconstructions)
        )

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        total_vae_loss = reconstruction_loss + kl_loss
        self.add_loss(total_vae_loss)

        return reconstructions # This layer should output the reconstructions

# Integrate the custom loss layer into the model
outputs_with_loss = VAELossLayer()([inputs, outputs, z_mean, z_log_var])

vae = Model(inputs, outputs_with_loss)
vae.compile(optimizer='adam') # No loss needed here, it's added internally by the layer

vae.summary()

history = vae.fit(
    data,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

plt.plot(history.history['loss'], label='Train Loss')
plt.title("VAE Training Loss")
plt.legend()
plt.show()

# Create decoder model separately
decoder_input = Input(shape=(latent_dim,))
_decoded_h = decoder_h(decoder_input)
_decoded_output = decoder_output(_decoded_h)

decoder = Model(decoder_input, _decoded_output)

# Generate new samples
z_sample = np.random.normal(size=(5, latent_dim))
generated_data = decoder.predict(z_sample)

print("Generated Data:\n", generated_data)

