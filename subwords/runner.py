from ml_classifier.decision_tree import get_config
import pandas as pd
from pkg_resources import resource_filename
from subwords.embedding import embeddings
from transformers import BertTokenizer

if __name__ == "__main__":
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_file_path']['path'])
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])

    df = pd.read_csv(feature_neg_file_path, sep=';')
    result = df.drop("Unnamed: 0", axis=1)
    for index, row in result.iterrows():
        neg_text_input = row['test_input']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(neg_text_input)
        print(neg_text_input)
        if len(tokens) < 510:
            try:
                embeddings(neg_text_input)
            except ValueError:
                print('value error')

    df = pd.read_csv(feature_pos_file_path, sep=',')
    for index, row in df.iterrows():
        pos_text_input = row['text']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(pos_text_input)
        if len(tokens) < 510:
            try:
                embeddings(pos_text_input)
            except ValueError:
                print('value error')

