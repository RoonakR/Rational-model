import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, Dropout, TimeDistributed, LSTM, Add
from keras_layer_normalization import LayerNormalization
import keras.backend as K


class Multi_head_time_attention(Layer):
    """docstring for Multi_head_time_attention"""

    def __init__(self, input_dims=76, multiple_weights=False, activation=None, **kwargs):
        super(Multi_head_time_attention, self).__init__(**kwargs)
        self.activation = activation
        self.multiple_weights = multiple_weights
        self.input_dims = input_dims

    def build(self, input_shape):
        """
        input shape: N * T * F

        Weight Matrix:
            T * F * Output_dim
        """
        if self.multiple_weights:
          self.WQ = LSTM(self.input_dims, return_sequences=True)  
          self.WK = LSTM(self.input_dims, return_sequences=True)  
          self.WV = LSTM(self.input_dims, return_sequences=True)  
        else:
          self.WQ = TimeDistributed(Dense(input_shape[-1]))
          self.WK = TimeDistributed(Dense(input_shape[-1]))
          self.WV = TimeDistributed(Dense(input_shape[-1]))

        super(Multi_head_time_attention, self).build(input_shape)


    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, important_rate=None,training=None):
        if self.multiple_weights:
          q = self.WQ(inputs, mask=important_rate)
          k = self.WK(inputs, mask=important_rate)
          v = self.WV(inputs, mask=important_rate)
          d_k = q.shape.as_list()[2]
          weights = K.batch_dot(q, k, axes=[2, 2])
          normalized_weights = K.softmax(weights / np.sqrt(d_k))
          output = K.batch_dot(normalized_weights, v)
        else:
          q = self.WQ(inputs)
          k = self.WK(inputs)
          v = self.WV(inputs)
          d_k = q.shape.as_list()[1]

          weights = K.dot(q, K.transpose(k))
          normalized_weights = K.softmax(weights / np.sqrt(d_k))
          output = K.dot(normalized_weights, v)
        
        return output


    def compute_output_shape(self, input_shape):
        return input_shape


def attention_block(inputs,input_dims=76, important_rate=None, multiple_weights=True):
    attention = Multi_head_time_attention(input_dims=input_dims, multiple_weights=multiple_weights)(inputs=inputs, important_rate=important_rate)
    attention = Dropout(0.5)(attention)
    add_merge = Add()([inputs, attention])
    return LayerNormalization()(add_merge), attention, add_merge

