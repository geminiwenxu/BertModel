from classifier.bert_model import BertBinaryClassifier
from classifier.prepare_data import prepare_data
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import BertTokenizer
import json
import yaml
from pkg_resources import resource_filename

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
BATCH_SIZE = 50
EPOCHS = 1
bert_clf = BertBinaryClassifier()
bert_clf.to(device)
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=3e-6)
train_dataloader, train_data, dev_dataloader, dev_data, dev_y = prepare_data(BATCH_SIZE)


def main():
    # training model
    for epoch_num in range(EPOCHS):
        bert_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            probas = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas, labels).to(device)
            train_loss += batch_loss.item()
            bert_clf.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print('Epoch: ', epoch_num + 1)
            print(
                "\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))

    # evaluate model
    bert_clf.eval()
    bert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(dev_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss().to(device)
            loss = loss_func(logits, labels)
            numpy_logits = logits.cpu().detach().numpy()
            bert_predicted += list(numpy_logits[:, 0] > 0.5)
            all_logits += list(numpy_logits[:, 0])

    print(classification_report(dev_y, bert_predicted))


if __name__ == '__main__':
    main()
