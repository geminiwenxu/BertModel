from keras.models import Sequential
from keras.layers import Dense
from pkg_resources import resource_filename
import yaml
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout
from keras.constraints import maxnorm


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def prepare_data(neg_path, pos_path):
    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()

    Y = [0] * len(neg_lines) + [1] * len(pos_lines)
    X = []
    for i in neg_lines:
        obj = json.loads(i)
        feature = np.array(obj['feature'])
        feature = np.nan_to_num(feature.astype(np.float32))
        for i in range(len(feature)):
            if feature[i] > 205:
                feature[i] = 1
        X.append(feature)
    for i in pos_lines:
        obj = json.loads(i)
        feature = np.array(obj['feature'])
        feature = np.nan_to_num(feature.astype(np.float32))
        for i in range(len(feature)):
            if feature[i] > 205:
                feature[i] = 1
        X.append(feature)
    arr_x = np.array(X)
    arr_y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.2, shuffle=True, stratify=arr_y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def nn(X_train, X_test, y_train, y_test):
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model = Sequential()
    model.add(Dense(100, input_dim=len(X_train[0, :]), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(80, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(40, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(4, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=False, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_bool = []
    for i, predicted in enumerate(y_pred):
        if predicted[0] > 0.5:
            y_pred_bool.append(1)
        else:
            y_pred_bool.append(0)

    print(classification_report(y_test, y_pred_bool))


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    X_train, X_test, y_train, y_test = prepare_data(neg_path, pos_path)
    nn(X_train, X_test, y_train, y_test)


