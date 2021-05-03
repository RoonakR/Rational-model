import tensorflow as tf

if tf.__version__ >= '2.0':
    from tensorflow.keras.layers import Layer
else:
    from keras.layers import Layer


# Multiple Centers 
class Probability_CLF_Mul(Layer):
    """docstring for Probability_CLF"""
    def __init__(self, output_dim, num_centers=1, non_trainable=0, activation=None, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        self.activation = activation
        super(Probability_CLF_Mul, self).__init__(**kwargs)

    def build(self,input_shape):
        self.centers = {}
         
        for idx in range(self.output_dim):
            self.centers[idx] = []
            
            
            if idx in range(self.non_trainable):
                trainable = False
            else:
                trainable = True
            for c in range(self.num_centers):
                W = self.add_weight(name='center%d_%d'%(idx, c), shape=(input_shape[1], ), initializer='uniform', trainable=True)
                self.centers[idx].append(W)
            
        super(Probability_CLF_Mul, self).build(input_shape)      

    def call(self, x, training=None):
        logits = []
        re_logits = []
        # Fixed Sigma
        sigma = 1.0

        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):

                G.append(self.gaussian_activation(tf.math.squared_difference(x, self.centers[idx][c]), sigma))
    
            G = tf.stack(G,axis=1)
            
            
            P = tf.reduce_sum(G,axis=1) / (tf.reduce_sum(G,axis=1) + self.num_centers - tf.reduce_max(G,axis=1) * self.num_centers )
            
            logits.append(P)

            
        
        logits = tf.stack(logits, axis=1)
        re_logits = logits
        if self.activation is not None:
            re_logits = self.activation(logits)
        

        return logits

    def gaussian_activation(self, x, sigma=None):
        sigma = 1. if sigma == None else sigma
        return tf.exp(-tf.reduce_sum(x, axis=1) / (2. * sigma * sigma)) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)