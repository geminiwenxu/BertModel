from ml_classifier.decision_tree import get_config
import pandas as pd
from pkg_resources import resource_filename
from subwords.embedding import embeddings
from transformers import BertTokenizer
import re


def sentence_em():
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_file_path']['path'])
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])
    feature_test_file_path = resource_filename(__name__, config['feature_test_file_path']['path'])

    df = pd.read_csv(feature_test_file_path, sep=';')
    result = df.drop("Unnamed: 0", axis=1)
    for index, row in result.iterrows():
        neg_text_input = row['test_input']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(neg_text_input)
        # print(neg_text_input)
        sen_split = re.findall(r'\w+|\S+', neg_text_input)
        low_sen_split = [i.lower() for i in sen_split]
        # low_sen_split.remove('.')
        # print(low_sen_split)
        if len(tokens) < 510:
            try:
                sentence_embeddings, tokenized_text = embeddings(neg_text_input)
                # tokenized_text.remove('.')
                common_tokens = list(set(tokenized_text).intersection(low_sen_split))
                diff_tokens = list(set(tokenized_text) - set(low_sen_split))
                # print(sentence_embeddings)
                # print(tokenized_text)
                # print("common tokens: ", common_tokens)
                # print("different tokens: ", diff_tokens)
            except ValueError:
                print('value error')
    return sentence_embeddings, tokenized_text, common_tokens, diff_tokens

    # df = pd.read_csv(feature_pos_file_path, sep=',')
    # for index, row in df.iterrows():
    #     pos_text_input = row['text']
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     tokens = tokenizer.tokenize(pos_text_input)
    #     if len(tokens) < 510:
    #         try:
    #             embeddings(pos_text_input)
    #         except ValueError:
    #             print('value error')


if __name__ == "__main__":
    sentence_em()
