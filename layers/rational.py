import tensorflow as tf
from keras_layer_normalization import LayerNormalization
from keras.layers import Lambda, Layer, Dense, Dropout, TimeDistributed
from keras.activations import sigmoid
import keras.backend as K
import numpy as np


def loss_func(z, probs):
    return -(z * tf.log(probs + 1e-8) + (1. - z) * tf.log(1. - probs + 1e-8))


class Generator_time(Layer):
    """docstring for Generator"""

    def __init__(self, **kwargs):
        super(Generator_time, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input shape: N * T * F

        Weight Matrix:
            T * F * Output_dim
        """
        self.hidden1 = Dense(200, activation='relu')
        self.hidden2 = Dense(1)
        self.drop = Dropout(0.3)
        self.ffn_norm = LayerNormalization(epsilon=1e-6)
        self.vals = Dense(1)

        super(Generator_time, self).build(input_shape)

    def call(self, x):
        ffn = self.hidden1(x)
        ffn = self.hidden2(ffn)
        ffn_dropout = self.drop(ffn)
        ffn_norm = self.ffn_norm(ffn_dropout)
        vals = self.vals(ffn_norm)
        probs = sigmoid(vals)
        uniform = tf.contrib.distributions.Uniform()
        samples = uniform.sample(tf.shape(probs))

        z = tf.to_float(tf.less(samples, probs))
        loss = Lambda(lambda x: loss_func(x, probs))(z)

        return [z, loss]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]


class get_generator(Layer):
    """docstring for Generator"""

    def __init__(self, **kwargs):
        self.temperature = tf.Variable(1e-4, trainable=False)
        self.sparsity = tf.Variable(100, trainable=False)
        self.mask = tf.Variable(np.zeros([50, 24, 2]), trainable=False, dtype=tf.float32)
        super(get_generator, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input shape: N * T * F

        Weight Matrix:
            T * F * Output_dim
        """
        self.hidden = TimeDistributed(Dense(2))
        self.droput = Dropout(0.3)
        self.batch_size = input_shape[0]
        super(get_generator, self).build(input_shape)

    def call(self, x, training=False):
        training = K.learning_phase()
        outputs = self.hidden(x)
        outputs = self.droput(outputs, training=K.learning_phase())
        self.outputs = outputs
        z_hard = Lambda(lambda x: gumbel_softmax(x, self.temperature, mask=None, hard=True))(outputs)
        self.loss = None
        return K.in_train_phase(
            slice_output(Lambda(lambda x: gumbel_softmax(x, self.temperature, mask=self.mask))(outputs)),
            slice_output(z_hard), training=training)

    def get_loss(self):
        return self.loss

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return U  # -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, mask=None, hard=False):
    if not hard:
        # logits = tf.nn.softmax(logits)
        gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits)) * mask[:K.shape(logits)[0]]
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    else:
        y = logits  # + sample_gumbel(tf.shape(logits))
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
        # y = y_hard
    return y


def slice_output(x):
    return tf.expand_dims(x[:, :, 1], -1)
