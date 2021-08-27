#!/usr/bin/env python3
"""
module to create sparse autoencoder function
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function to create a sparse autoencoder
    Arguments:
    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each hidden
        layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    - latent_dims is an integer containing the dimensions of the latent space
        representation
    Returns:
    - encoder is the encoder model
    - decoder is the decoder model
    - auto is the full autoencoder model
    - lambtha is the regularization parameter used for L1 regularization on the
        encoded output
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0], activation='relu')(input_encoder)

    for layer in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[layer], activation='relu')(
            encoded)
    regularizer = keras.regularizers.l1(lambtha)
    encoder_out = keras.layers.Dense(latent_dims, activation='relu',
                                     activity_regularizer=regularizer)(encoded)
    input_decoder = keras.layers.Input(shape=(latent_dims,))
    decoded = input_decoder
    for layer in range(len(hidden_layers) - 1, - 1, - 1):
        decoded = keras.layers.Dense(hidden_layers[layer], activation='relu')(
            decoded)
    decoded = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(input_encoder, encoder_out)
    decoder = keras.Model(input_decoder, decoded)
    auto = keras.Model(input_encoder, decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
