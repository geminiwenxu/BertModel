import pandas as pd
import torch
import yaml
from pkg_resources import resource_filename
from sklearn.metrics import classification_report
from transformers import BertTokenizer
import numpy as np
from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model import logger


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def prediction(model_path, file_path):
    config = get_config('/../config/config.yaml')
    class_names = config['class_names']
    dropout_ratio = config['dropout_ratio']
    if torch.cuda.is_available():
        logger.info("CUDA is available, setting up Tensors to work with CUDA")
        device = torch.device("cuda")
    else:
        logger.info("CUDA is NOT available, setting up Tensors to work with CPU")
        device = torch.device("cpu")
    df = pd.read_csv(file_path, sep=';',
                     header=None, index_col=False, names=['actual', 'text', 'lang', 'source'])
    model = SentimentClassifier(len(class_names), dropout_ratio)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage).module.state_dict())
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    predict = []
    L = []
    for index, row in df.iterrows():
        encoded_review = tokenizer.encode_plus(row['text'],
                                               add_special_tokens=True,
                                               max_length=160,
                                               return_token_type_ids=False,
                                               padding='max_length',
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        input_ids = encoded_review['input_ids'].to(device)

        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        pred = np.float64(prediction.cpu().detach().numpy()[0])
        actual = np.float64(row['actual'])

        if pred == actual:
            print(pred, actual)
            print(type(pred), type(actual))
            L.append({"actual": row['actual'], "text": row['text'], "prediction": pred})

        predict.append(prediction)
    report = classification_report(df.actual.to_list(), predict, target_names=class_names)
    print(report)
    correct_classified = pd.DataFrame(L)
    correct_classified.to_csv('correct_classification.csv', index=False)
    return None


if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    model_path = resource_filename(__name__, config['model_path']['path'])
    print(model_path)
    log_path = resource_filename(__name__, config['logs']['path'])
    print(log_path)
