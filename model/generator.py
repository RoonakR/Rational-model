import keras.backend as K
from keras.layers import Dense, Input, Flatten, Lambda, Multiply, LSTM
from keras.models import Model
from keras_pos_embd import TrigPosEmbedding

from layers.rational import get_generator
from layers.attention import attention_block
from layers.residual import residual_block
from layers.probability_clf_mul import Probability_CLF_Mul
import keras
from model.losses import *


def model_generator(params, retrun_generator=False, input_shape=(48, 76), output_shape=2):
    use_lstm = params['use_lstm']
    print('model information'.center(50, '-'))
    for key, value in params.items():
        print('{0:<10} --- '.format(key), value)
    print('End'.center(50, '-'))

    inputs = Input(shape=input_shape)
    hidden = inputs
    hidden_dim = 8

    ''' Position Embedding'''
    if params['PE'] and use_lstm:
        hidden = TrigPosEmbedding(output_dim=hidden_dim, mode=TrigPosEmbedding.MODE_ADD)(hidden)

    ''' Rational Block'''
    for i in range(params['Rational']):
        generator = get_generator()
        important_rate = generator(hidden)
        K.stop_gradient(important_rate)
        hidden = Lambda(lambda x: keras.layers.multiply([x[0], x[1]]))([important_rate, hidden])
        if retrun_generator:
            gener_output = hidden
        hidden = LSTM(hidden_dim, return_sequences=use_lstm)(hidden, mask=K.greater(important_rate, 0.5))


    ''' Attention Block'''
    for i in range(params['Attention']):
        hidden, attention, attention_out = attention_block(hidden, hidden_dim, multiple_weights=use_lstm)

    ''' Residual Block'''
    for i in range(params['Residual']):
        hidden = residual_block(hidden, hidden_dim)

    if use_lstm:
        try:
            hidden = Flatten()(hidden)
        except ValueError:
            pass

    ''' Hidden Layers '''
    for i in range(params['Hidden']):
        if not use_lstm:
            if i == params['Hidden'] - 1:
                hidden = LSTM(200, activation='relu')(hidden)
            else:
                hidden = LSTM(200, activation='relu', return_sequences=True)(hidden)
        else:
            hidden = Dense(200, activation='relu')(hidden)

    if params['PNN']:
        outputs = Probability_CLF_Mul(output_shape)(hidden)
    else:
        outputs = Dense(output_shape, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=outputs)

    def get_loss_func():
        name = params['Loss']
        if params['Rational']:
            if name == 'focal':
                return integrated_loss(important_rate, generator.get_loss(),params['sparsity'])
            elif name == 'gener':
                return gener_loss(important_rate, generator.get_loss())
            elif name == 'crossen':
                return 'binary_crossentropy'
            elif name == 'mse':
                return 'mse'
        else:
            if name == 'focal':
                return binary_focal_loss()
            elif name == 'gener':
                return None
            elif name == 'crossen':
                return 'binary_crossentropy'
            elif name == 'mse':
                return 'mse'

    model.compile(loss=get_loss_func(), optimizer=keras.optimizers.Adam(learning_rate=params['lr']), metrics=['accuracy'])

    if retrun_generator:
        model_attention = Model(inputs=inputs, outputs=attention) if params['Attention'] else None
        model_attention_out = Model(inputs=inputs, outputs=attention_out) if params['Attention'] else None
        model_time = Model(inputs=inputs, outputs=important_rate)
        return model, model_attention, model_time, model_attention_out

    return model


