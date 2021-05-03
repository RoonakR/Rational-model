from keras.layers import Dense, Dropout, Add
from keras_layer_normalization import LayerNormalization

def residual_block(inputs, output_dim):
    hidden = Dense(output_dim, activation='relu')(inputs)
    hidden = Dense(output_dim)(hidden)
    hidden = Dropout(0.5)(hidden)
    add_merge = Add()([inputs, hidden])
    return LayerNormalization()(add_merge)

