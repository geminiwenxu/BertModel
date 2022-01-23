import json
from itertools import combinations
from pkg_resources import resource_filename
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
    feature_path = resource_filename(__name__, config['feature_file_path']['path'])

    feature_names = config['feature_names']
    ls_features = ['depH', 'NDW', 'Q', 'dpd', 'advpd', 'RR', 'preppd', 'adjpd', 'btac1', 'G', 'imbG', 'wG', 'DDEmu',
                   'btH', 'lmbd', 'ASL', 'btadc10', 'npd',
                   'RRR', 'btadc10', 'Lradc', 'btrac', 'btac4', 'deprac', 'btadc2', 'btadc10', 'UG', 'btac1', 'btac3',
                   'btac2', 'btac10', 'alpha',
                   'Lradc', 'cG', 'btrac', 'btadc9', 'btac4', 'DDEradc', 'deprac', 'depG']

    ls_index = []  # 0.62
    for i in ls_features:
        index = feature_names.index(i)
        ls_index.append(index)
    ls_first_feature = [2, 4, 7, 9, 15, 28, 44, 45, 51, 52, 54, 56, 57, 58, 61, 84, 85, 86, 87, 90, 91, 92, 94,
                        106]  # 0.61
    best_features_combination = [2, 15, 52, 106]  # 0.59
    another_best_combination = [0, 2, 3, 6, 8, 10, 16, 18, 19, 21, 22, 26, 32, 34, 35, 45, 48, 52, 87, 106, 113, 114,
                                115, 117, 118, 124, 126, 127]  # 0.61
    with open(neg_path) as f:
        neg_lines = f.readlines()
    with open(pos_path) as f:
        pos_lines = f.readlines()
    Y = [0] * len(neg_lines) + [1] * len(pos_lines)

    with open(feature_path) as f:
        lines = f.readlines()

    X = []
    for i in lines:
        obj = json.loads(i)
        feature_vec = np.array(obj['feature'])
        features = [feature_vec[i] for i in
                    another_best_combination]  # using all useful features from decision tree or single feature list skip this line
        feature = np.asarray(features)  # feature_vec instead of features
        feature = np.nan_to_num(feature.astype(np.float32))
        X.append(feature)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = SVC(gamma='auto', cache_size=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    class_names = ['correct_classified', 'misclassified']
    print(classification_report(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    repdf = pd.DataFrame(report_dict).round(2).transpose()
    repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    save_path = resource_filename(__name__, config['svm_save_path']['path'])
    repdf.to_csv(
        save_path + "best combination of features.csv",
        index=False)


if __name__ == "__main__":
    main()
