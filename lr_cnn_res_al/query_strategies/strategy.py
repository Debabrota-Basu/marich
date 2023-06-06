import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import LogisticRegression, ResNet, CNN, CNN_res, CNN_img, ResNet_small, ResNet18

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False
        print("Loading last best model")
        if self.net.params["net"] == "CNN_img":
            self.net.net = CNN_img()
        elif self.net.params["net"] == "CNN":
            self.net.net = CNN()
        elif self.net.params["net"] == "LogReg":
            self.net.net = LogisticRegression()
        elif self.net.params["net"] == "ResNet_small":
            self.net.net = ResNet18()
        self.net.net.load_state_dict(torch.load("./models/"+self.net.params["net"]+"_"+self.net.params["algo"]+"_"+self.net.params["data"]+".pt"))

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

