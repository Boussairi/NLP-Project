import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from transformers import BertTokenizer


class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        f1 = f1_score(labels, pred)

        return {"accuracy": accuracy,"f1_score":f1}

    def labels(x):
        if x == 0:
            return 0
        else:
            return 1


    if __name__ == '__main__':
        path = r'train\train.Ar.csvv'
        path_test = r'test/task_A_En_test.csv'

        df = pd.read_csv(path)
        test = pd.read_csv(path_test)
        df = df.dropna(subset=['tweet'])

        train = df

        train_tweets = train['tweet'].values.tolist()
        train_labels = train['sarcastic'].values.tolist()
        test_tweets = test['text'].values.tolist()

        train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, 
                                                                            test_size=0.1,random_state=42,stratify=train_labels)
        model_name = 'detecting-Sarcasm'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=2,
                                                loss_function_params={"weight": [0.75, 0.25]})
