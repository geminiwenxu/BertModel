from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from pkg_resources import resource_filename
import yaml
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from keras import backend as K


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
        X.append(feature)
    for i in pos_lines:
        obj = json.loads(i)
        feature = np.array(obj['feature'])
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test= np.array(y_test)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    return X_train, X_test, y_train, y_test


def nn(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(12, input_dim=135, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X_train, y_train, epochs=150, batch_size=10)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))

    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_bool))

if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    X_train, X_test, y_train, y_test = prepare_data(neg_path, pos_path)
    nn(X_train, X_test, y_train, y_test)