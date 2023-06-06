import numpy as np
import torch
from torchvision import datasets, transforms
from nets import BertClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class basic_dataset(Dataset):
    def __init__(self, text, Y = None):
        self.text = text
        self.Y = Y

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if type(idx) in [list, np.ndarray]:
            x = [self.text[i] for i in idx]
        else:
            x = self.text[idx]
        y = self.Y[idx]
        return x, y

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset_2(Dataset):

    def __init__(self, text, labels):

        self.labels = torch.LongTensor(labels)
        self.text = [tokenizer(x, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for x in text]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        if type(idx) in [list, np.ndarray]:
            x = [self.text[i] for i in idx]
        else:
            x = self.text[idx]
        return x
    
    def get_data(self,idx):
        if type(idx) in [list, np.ndarray]:
            x = torch.stack([self.text[i]["input_ids"] for i in idx])
        else:
            x = self.text[idx]["input_ids"]
        return x

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    def get_dataset(self, idx):
        if type(idx) in [list, np.ndarray]:
            x = [self.text[i] for i in idx]
        else:
            x = self.text[idx]
        y = self.labels[idx]
        return basic_dataset(x,y)
    
    def get_label(self,idx):
        y = self.labels[idx]
        return y
    
    def get_data_label_loader(self, idx, batch_size = 8):
        
        if type(idx) in [list, np.ndarray]:
            x = [self.text[i] for i in idx]
        else:
            x = self.text[idx]
        y = self.labels[idx]
        
        return_set = basic_dataset(x,y)
        return torch.utils.data.DataLoader(return_set, batch_size = batch_size)


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        x = [self.X_train[i] for i in labeled_idxs]
        y = [self.Y_train[i] for i in labeled_idxs]
        return labeled_idxs, self.handler(x, y)
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        x = [self.X_train[i] for i in unlabeled_idxs]
        y = [self.Y_train[i] for i in unlabeled_idxs]
        return unlabeled_idxs, self.handler(x, y)
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

def get_BERT(handler):
    with open('ag_train_unbal_x.pkl', 'rb') as f:
        unlab_x = pickle.load(f)
    with open('ag_train_unbal_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)

    test = pd.read_csv("./bert_data/bbc/bbc_train.csv")
    train, test =train_test_split(test, test_size = 290)
    labels_dict = {np.unique(test["Category"])[i]:i for i in range(5)}
    test_text = test["Text"]
    test_labels = np.array(test["Category"].map(labels_dict))
    return Data(unlab_x, unlab_y, test_text, torch.LongTensor(test_labels), handler)