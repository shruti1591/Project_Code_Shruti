# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense,Conv1D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from SFO import*

class SparseAutoencoder(tf.keras.Model):
    def __init__(self, encoding_dim,sparsity_factor):
        super(SparseAutoencoder, self).__init__()
        self.encoder = tf.keras.layers.Dense(encoding_dim, activation='sigmoid',
                            activity_regularizer=regularizers.l1(sparsity_factor))
        self.decoder = tf.keras.layers.Dense(100, activation='sigmoid')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def text_model(inputss):

    input_shape=Input(inputss)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_shape)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    
    encoding_dim = 10  
    sparsity_factor = 1e-5  
    autoencoder = SparseAutoencoder(encoding_dim,sparsity_factor)(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(autoencoder)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x=Flatten()(x)
    x=Dense(6,activation='softmax')(x)

    model=Model(input_shape,x)
    model.compile(loss = 'CategoricalCrossentropy', optimizer = SFO(),metrics=['Accuracy'])
    return model
    
def abliation_text(inputss):

    input_shape=Input(inputss)   
    encoding_dim = 10  # Set the size of the encoding layer
    sparsity_factor = 1e-5  # Adjust the sparsity factor
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_shape)
    x = SparseAutoencoder(encoding_dim,sparsity_factor)(x)
    x=Flatten()(x)
    x=Dense(6,activation='softmax')(x)

    model=Model(input_shape,x)
    model.compile(loss = 'CategoricalCrossentropy', optimizer = SFO(),metrics=['Accuracy'])
    return model



# ---------------------audio-------------
from keras.layers import Dense
from keras import Model,Input
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import  LSTM, Bidirectional,Flatten
from tensorflow.keras.layers import Layer
import keras.backend as K
from keras import initializers, layers
# from keras.saving import register_keras_serializable
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout,Concatenate, Bidirectional, GRU, GlobalAveragePooling1D, Dense
import keras
import numpy as np

"capsule layer"
  
# Define the Squash activation function
def squash(vectors, axis=-1):
    norm = tf.norm(vectors, axis=axis, keepdims=True)
    squared_norm = norm**2
    squash_factor = squared_norm / (1 + squared_norm)
    unit_vector = vectors / norm
    output = squash_factor * unit_vector
    return output


# @tf.keras.utils.register_keras_serializable()
class PrimaryCapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv2d = layers.Conv1D(num_capsules * capsule_dim, kernel_size, strides=strides, padding=padding,name='DigitCapsuleLayer')
    def call(self, inputs):
        outputs = self.conv2d(inputs)
        outputs = squash(outputs)
        return outputs
    def get_config(self):
      config = super(PrimaryCapsuleLayer,self).get_config()
      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)



# @tf.keras.utils.register_keras_serializable()
class DigitCapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, num_routing, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
        # Weight matrix for routing
        self.routing_weights = self.add_weight(shape=(1, num_capsules, 1, capsule_dim, capsule_dim),
                                              initializer='random_normal',name='DigitCapsuleLayer',
                                              trainable=True)
    def call(self, inputs):
        # Expand dimensions to match shape for routing
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 1), 4)
        # Tile the routing weights to match the batch size
        routing_weights_expanded = tf.tile(self.routing_weights, [tf.shape(inputs)[0], 1, tf.shape(inputs)[1], 1, 1])
        # print(routing_weights_expanded.shape)
        # Perform dynamic routing
        for _ in range(self.num_routing):
            # Update coupling coefficients using softmax
            coupling_coefficients = tf.nn.softmax(routing_weights_expanded, axis=2)
            
            weighted_sum = tf.reduce_sum(coupling_coefficients * inputs_expanded, axis=-1, keepdims=True)
            # Apply squash activation
            outputs = squash(weighted_sum)
            # Update routing weights
            delta_routing_weights = tf.reduce_sum(inputs_expanded * outputs, axis=3, keepdims=True)
            routing_weights_expanded += delta_routing_weights
        return outputs
    
    
    def get_config(self):
      config = super(DigitCapsuleLayer,self).get_config()
      # config.update({'num_capsules':self.num_capsules,'routing_weights':self.routing_weights,
      #               'num_routing':self.num_routing ,'capsule_dim':self.capsule_dim})
      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)    
    
    


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, 1), initializer='random_normal', trainable=True,name='W')
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True,name='b')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


def BiG_ACN(input_shape_1):
    
    input_layer = Input(input_shape_1)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_layer)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Bidirectional(GRU(37, activation='relu', return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x=AttentionLayer()(x)
    x = Conv1D(filters=10, kernel_size=1, activation='relu')(x)
    
    x = PrimaryCapsuleLayer(num_capsules=1, capsule_dim=1, kernel_size=1, strides=2, padding='valid')(x)   
    x = DigitCapsuleLayer(num_capsules=1, capsule_dim=1, num_routing=3)(x)
    x=Flatten()(x)
    output_layer = Dense(6, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss = 'CategoricalCrossentropy', optimizer =SFO(),metrics=['Accuracy'])
    
    return model


def without_BiG_ACN(input_shape_1):
    
    input_layer = Input(input_shape_1)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_layer)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Bidirectional(GRU(37, activation='relu', return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=10, kernel_size=1, activation='relu')(x)
    
    x = PrimaryCapsuleLayer(num_capsules=1, capsule_dim=1, kernel_size=1, strides=2, padding='valid')(x)   
    x = DigitCapsuleLayer(num_capsules=1, capsule_dim=1, num_routing=3)(x)
    x=Flatten()(x)
    output_layer = Dense(6, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss = 'CategoricalCrossentropy', optimizer =SFO(),metrics=['Accuracy'])
    
    return model


