import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from scipy.stats import entropy
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup


class Net:
    def __init__(self, net, params, device, y_num = None):
        self.net = net
        self.params = params
        self.device = device
        self.y_num = y_num
        self.clf = self.net().to(self.device)
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        # self.clf = self.net().to(self.device)
        self.clf.train()
        tr = int(0.8*len(data))
        vl = len(data) - tr
        data, val = random_split(data, [tr, vl])
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        val_loader = DataLoader(data, shuffle=True, **self.params['test_args'])
        total_steps = len(loader)*n_epoch
        if self.params["name"] == "BERT":
            optimizer = optim.AdamW(self.clf.parameters(), lr = self.params["optimizer_args"]["lr"], eps = 1e-8 )
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        losses = []
        val_losses = []
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            loss_e = 0
            tr_acc = 0
            total = 0
            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(loader):
                # x, y = x.to(self.device), y.to(self.device)
                self.clf.train()
                optimizer.zero_grad()
                out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                cost = nn.CrossEntropyLoss()
                loss = cost(out.to(self.device), y.to(self.device))
                tr_acc += torch.sum(torch.argmax(out.to(self.device), dim = 1) == y.to(self.device))
                total += len(x)
                del x
                del out
                del y
                del e1
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_e += float(loss)
            del loss
            losses.append(loss_e/len(loader))
            tr_acc = tr_acc/total
            print("Train accuracy = ", tr_acc)

            loss_v = 0
            for batch_idx, (x, y, idxs) in enumerate(val_loader):
                # x,y = x.to(self.device), y.to(self.device)
                self.clf.eval()
                out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                cost = nn.CrossEntropyLoss()
                loss_val = float(cost(out.to(self.device), y.to(self.device)))
                loss_v += loss_val
                del x
                del y
                del out
                del e1
            val_losses.append(loss_v/len(val_loader))
            if loss_v/len(val_loader)<=val_losses[-1]:
                print("saving model")
                torch.save(self.clf.state_dict(), "./models/"+self.params["net"]+"_"+self.params["algo"]+"_"+self.params["data"]+".pt")
                    
            

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.labels.dtype)
        target_preds = torch.zeros(len(data), dtype=data.labels.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        if self.params["data"] == "BERT":
            if torch.cuda.is_available():
                model = BertClassifier().to(self.device)
                model.load_state_dict(torch.load("./target/bert.pt"))
            else:
                model = BertClassifier().cpu()
                model.load_state_dict(torch.load("./target/bert.pt", map_location=torch.device('cpu')))
        
        model.eval()
        
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                # x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                target_out, _ = model(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                pred = out.max(1)[1]
                target_pred = target_out.max(1)[1]
                preds[idxs] = pred.cpu()
                target_preds[idxs] = target_pred.cpu()
        agreement = torch.mean(preds == target_preds, dtype = torch.float).item()*100
        classes = len(np.unique(data.labels))
        y_t_c = []
        y_e_c = []
        for i in range(classes):
            y_t_c.append(sum(target_preds == i)+1)
            y_e_c.append(sum(preds == i)+1)
        y_t_p = np.array(y_t_c)/(len(data)+classes)
        y_e_p = np.array(y_e_c)/(len(data)+classes)
        kl_div = entropy(y_t_p, y_e_p)
        return preds, agreement, kl_div
    
    def predict_prob(self, data):
        self.clf.eval()
        if self.y_num != None:
            probs = torch.zeros([len(data), self.y_num])
        else:
            probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        if self.y_num != None:
            probs = torch.zeros([len(data), self.y_num])
        else:
            probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        if self.y_num != None:
            probs = torch.zeros([len(data), self.y_num])
        else:
            probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x["input_ids"].squeeze(dim = 1).to(self.device), x["attention_mask"].squeeze(dim = 1).to(self.device))
                embeddings[idxs] = e1.squeeze().cpu()
        return embeddings
    
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer, pooled_output
    
    def get_embedding_dim(self):
        return 768