import numpy as np
from keras.utils import to_categorical
from utils.dataloader import load_all_data, split_dataframe
from sklearn.model_selection import train_test_split
from utils.util import get_params, get_combination, get_result

# Hyper Parameters
VERBOSE = 0
EPOCHS = 100
MULTIPLE_WEIGHTS = True

# Load the data
df = load_all_data()
data_unlabel, data, label, body_temp = split_dataframe(df)
data = data.reshape(data.shape[0], 24, 8)
label_enc = to_categorical(np.array(label), 2)
X_train, X_test, y_train, y_test = train_test_split(data, label_enc, test_size=0.33, random_state=42, stratify=label_enc)

params = get_params()
combinations = get_combination(params)
get_result(X_train, y_train, X_test, y_test, combinations, verbose=1)