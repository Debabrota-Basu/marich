from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class BERT_Handler(Dataset):

    def __init__(self, text, labels):

        self.labels = torch.LongTensor(labels)
        self.text = [tokenizer(x, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for x in text]
    def classes(self):
        return self.labels
    
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        if type(idx) in [list, np.ndarray]:
            x = [self.text[i] for i in idx]
        else:
            x = self.text[idx]
        return x
    
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y, idx