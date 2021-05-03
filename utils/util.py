import pickle
import itertools
from model.generator import model_generator
from utils.metrics import print_metrics_binary
from keras.callbacks import Callback
import numpy as np


def save_obj(obj, name):
    with open('history/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class History_callback(Callback):

    def __init__(self, model, model_time, X_test, y_test):
        self.model_time = model_time
        self.model = model
        self.history = {}
        self.history['auroc'] = []
        self.history['auprc'] = []
        self.history['rate'] = []
        self.selection_changes = []
        self.selection_stay = []
        self.pre_selection = None
        self.X_test = X_test
        self.y_test = y_test
        super().__init__()

    def on_epoch_end(self, batch, logs=None):
        res = print_metrics_binary(np.argmax(self.y_test, axis=1), self.model.predict(self.X_test).astype(np.float32), verbose=0)
        if self.model_time is not None:
            selection = self.model_time.predict(self.X_test)
            rate = np.sum(selection) / (self.X_test.shape[0] * self.X_test.shape[1])
            self.history['rate'].append(rate)
            if self.pre_selection is None:
                self.pre_selection = np.ones_like(selection)
            self.selection_changes.append(np.sum(self.pre_selection != selection))
            self.selection_stay.append(np.sum(self.pre_selection == selection))
        self.history['auroc'].append(res['auroc'])
        self.history['auprc'].append(res['auprc'])


def get_params():
    params = {
        # Model Architecture
        'Rational': [1],
        'Attention': [1],
        'Residual': [1],
        'Hidden': [1],
        'PNN': [False],
        'CW': [False],  # Class weights
        'PE': [True],  # Position Encoding
        'Loss': ['focal'],  # loss function, generator loss, focal loss, mse, categorical crossentropy
        'AU': [None],
        'Nor': [False],
        'use_lstm': [True],
        'lr': [0.0001],  # np.linspace(1e-4,1e-3,3),
        'sparsity': [0.001],  # np.linspace(1e-3,1e-3,1),
        'callback_decay': np.linspace(0.5, 0.9, 1),
        'callback_spar': np.linspace(0.5, 0.9, 1)
    }
    return params


def get_combination(params):
    flat = [[(k, v) for v in vs] for k, vs in params.items()]
    combinations = [dict(items) for items in itertools.product(*flat)]
    return combinations


def get_result(X_train, y_train, X_test, y_test, combinations, verbose=0, name=None):
    for c_idx, c in enumerate(combinations):
        print('Test model %d/%d' % (c_idx + 1, len(combinations)))
        if c['Rational'] == 0:
            model = model_generator(c, input_shape=(24, 8), output_shape=2, retrun_generator=False)
            model_time = None
        else:
            model, model_attention, model_time, model_attention_out = model_generator(c, input_shape=(24, 8),
                                                                                      output_shape=2,
                                                                                      retrun_generator=True)
        his_callback = History_callback(model, model_time, X_test, y_test)
        history = model.fit(X_train, y_train, verbose=verbose, epochs=50,
                            batch_size=50, callbacks=[his_callback])
        all_history = {}
        all_history.update(his_callback.history)
        all_history.update(history.history)
        if name is not None:
            save_obj(all_history, name)
        print_metrics_binary(np.argmax(y_test, axis=1), model.predict(X_test).astype(np.float32))
