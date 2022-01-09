import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
from keras.preprocessing.sequence import pad_sequences
import yaml
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
pos_feature_path = resource_filename(__name__, config['pos_feature_file_path']['path'])
neg_feature_path = resource_filename(__name__, config['neg_feature_file_path']['path'])


def prepare_data(BATCH_SIZE):
    pos_df = pd.read_json(pos_feature_path, lines=True)
    neg_df = pd.read_json(neg_feature_path, lines=True)
    for index, row in pos_df.iterrows():
        feature = row['feature']
        feature = [0 if v is None else v for v in feature]
        row['feature'] = feature

    for index, row in neg_df.iterrows():
        feature = row['feature']
        feature = [0 if v is None else v for v in feature]
        row['feature'] = feature

    pos_df['label'] = [1] * pos_df.shape[0]
    neg_df['label'] = [0] * neg_df.shape[0]
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

    train_data = [{'feature': feature, 'label': type_data} for feature in list(train_df['feature']) for type_data in
                  list(train_df['label'])]

    dev_data = [{'feature': feature, 'label': type_data} for feature in list(dev_df['feature']) for type_data in
                list(dev_df['label'])]

    train_features, train_labels = list(zip(*map(lambda d: (d['feature'], d['label']), train_data)))
    dev_features, dev_labels = list(zip(*map(lambda d: (d['feature'], d['label']), dev_data)))

    train_tokens = list(map(lambda t: [101] + t[:100] + [102], train_features))
    dev_tokens = list(map(lambda t: [101] + t[:100] + [102], dev_features))

    train_tokens_ids = train_tokens
    dev_tokens_ids = dev_tokens

    train_y = np.array(train_labels) == 1
    dev_y = np.array(dev_labels) == 1

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    dev_masks = [[float(i > 0) for i in ii] for ii in dev_tokens_ids]

    train_masks_tensor = torch.tensor(train_masks).long()
    dev_masks_tensor = torch.tensor(dev_masks).long()

    train_tokens_tensor = torch.tensor(train_tokens_ids).long()
    train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
    dev_tokens_tensor = torch.tensor(dev_tokens_ids).long()
    dev_y_tensor = torch.tensor(dev_y.reshape(-1, 1)).float()

    train_dataset = torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE,
                                                   num_workers=4)

    dev_dataset = torch.utils.data.TensorDataset(dev_tokens_tensor, dev_masks_tensor, dev_y_tensor)
    dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, sampler=dev_sampler, batch_size=BATCH_SIZE, num_workers=4)

    return train_dataloader, train_data, dev_dataloader, dev_data, dev_y


if __name__ == '__main__':
    prepare_data(1)
