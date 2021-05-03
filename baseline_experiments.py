import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from utils.dataloader import load_all_data, split_dataframe
from utils.metrics import print_metrics_binary 
from sklearn.linear_model import LinearRegression
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input
from layers.residual import residual_block

# Load the data
df = load_all_data()
data_unlabel, data, label, body_temp = split_dataframe(df)
data = data.reshape(data.shape[0], 24, 8)
label_enc = to_categorical(np.array(label), 2)
X_train, X_test, y_train, y_test = train_test_split(data, label_enc, test_size=0.33, random_state=42, stratify=label_enc)


print("LinearRegression:")
reg = LinearRegression().fit(X_train.reshape(X_train.shape[0], -1), y_train)
print_metrics_binary(np.argmax(y_test, axis=1), reg.predict(X_test.reshape(X_test.shape[0], -1)).astype(np.float32))

print("LSTM:")
inputs = Input(shape=(24,8))
hidden = residual_block(inputs, 8)
hidden = LSTM(200)(hidden)
outputs = Dense(2, activation='softmax')(hidden)
lstm_model = Model(inputs=inputs, outputs=outputs)
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=100, verbose=0)
print_metrics_binary(np.argmax(y_test, axis=1), lstm_model.predict(X_test).astype(np.float32))

print("Neural Network:")
nn_model = Sequential()
nn_model.add(Dense(200, input_shape=(24*8,)))
nn_model.add(Dense(2, activation='softmax'))
nn_model.compile(loss='categorical_crossentropy', optimizer='adam')
nn_model.fit(X_train.reshape(X_train.shape[0], -1), y_train, epochs=100, verbose=0)
print_metrics_binary(np.argmax(y_test, axis=1), nn_model.predict(X_test.reshape(X_test.shape[0], -1)).astype(np.float32))