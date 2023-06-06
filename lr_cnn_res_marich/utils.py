import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import time
from sklearn.cluster import KMeans
from scipy.stats import entropy
import torch.nn.functional as F
from torch.special import entr
import numpy as np
import gc
import random



class dataset(Dataset):
    def __init__(self, X, Y = None, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.X[idx].transpose(1,2,0))
        else:
            x = self.X[idx]
        if self.Y:
            y = self.Y[idx]
            return x, y
        else:
            return x
    
def conv_dataset(data):
  datas = []
  labels = []
  for d, l in data:
    datas.append(d)
    labels.append(torch.tensor(l))
  mydata = dataset(datas,labels)
  return mydata

class basic_dataset(Dataset):
    def __init__(self, X, Y = None, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        if self.Y != None:
            y = self.Y[idx]
            return x, y
        else:
            return x

class dataset2(Dataset):
    def __init__(self, X, Y = None, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        if self.Y != None:
            y = self.Y[idx]
            return x, y
        else:
            return x
    def get_data(self,idx):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        return x
    def get_label(self,idx):
        y = self.Y[idx]
        return y
    def get_dataset(self, idx):
        x = self.X[idx]
        y = torch.tensor(self.Y)[idx]
        return basic_dataset(x,y)

    def get_data_label_loader(self, idx, batch_size = 128):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        y = torch.tensor(self.Y)[idx]
        
        return_set = basic_dataset(x,y)
        return torch.utils.data.DataLoader(return_set, batch_size = batch_size)


def train(traindata, valloader, model, epochs, criterion, lr = 0.001, device = None, model_name = "logreg.pt", save = True, val_loss_list = [], model_type = "log_reg"):
    """
    traindata: Pytorch dataset
    valloader: Validation dataloader (Pytorch)
    model: Model to be trained
    epochs: Training epochs
    criterion: Loss function
    lr: Learning rate | Default: 0.001
    device: Any one from ["cuda", "cpu", "mps"]
    model_name: Name of the model to be saved | Default: "logreg.pt"
    save: Whether model is to be saved or not | Default: True
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    traindata = conv_dataset(traindata)
    trainloader = DataLoader(traindata, batch_size = 64, shuffle = True)
    train_loss_list = []
    if model_type == "log_reg":
        if lr == None:
            lr = 0.02
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = 0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.02, epochs=epochs, steps_per_epoch=len(trainloader))
    elif model_type == "cnn_mnist":
        if lr == None:
            lr = 0.015
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.95)
    elif model_type == "cnn_3c_small":
        if lr == None:
            lr = 0.02
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.85)
    elif model_type == "cnn_3c":
        if lr == None:
            lr = 0.02
        config = {                                                                                                                                                                                                          
            'batch_size': 16,
            'lr': 8.0505e-05,
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
        } 
        optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        amsgrad=config['amsgrad'], 
    )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(epochs):
        t1 = time.time()
        print("Epoch: ", epoch+1)
        model.train()
        train_loss = 0
        for  i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if model_type == "log_reg":
                outputs = model(inputs.reshape(inputs.shape[0],784))
            elif model_type == "cnn_mnist":
                outputs = model(inputs)
            elif model_type == "cnn_3c":
                outputs = model(inputs)
            elif model_type == "cnn_3c_small":
                outputs = model(inputs)
            del inputs
            if len(outputs.shape)<2:
                outputs = outputs.unsqueeze(dim = 0)
            loss = criterion(outputs,labels)
            del outputs
            loss.backward()
            optimizer.step()    
            train_loss+=float(loss.item())
            del loss
        train_loss_list.append(train_loss/len(trainloader))
        print("Train loss: ",train_loss/len(trainloader))
        model.eval()
        loss_val = 0
        for j, (ip,lbl) in enumerate(valloader):
            ip, lbl = ip.to(device), lbl.to(device)
            if model_type == "log_reg":
                op = model(ip.reshape(ip.shape[0],784))
            elif model_type == "cnn_mnist":
                op = model(ip)
            elif model_type == "cnn_3c_small":
                op = model(ip)
            elif model_type == "cnn_3c":
                op = model(ip)
            del ip
            val_loss = criterion(op, lbl)
            scheduler.step(val_loss.item())
            loss_val += float(val_loss.item())
            del val_loss, op
        val_loss_list.append(loss_val/len(valloader))
        print("Validation loss: ", val_loss_list[-1])
        print("Epoch time ----- ",time.time() - t1, " sec")
        if save:    
            if val_loss_list[-1]<=min(val_loss_list):
                print("validation loss minimum, saving model")
                torch.save(model.state_dict(), "./extracted_models/"+model_name)
    gc.collect()
    return train_loss_list, val_loss_list


def test(model, testloader, device = None, model_type = "log_reg"):
    """
    model: Model to be tested
    testedloader: Pytorch dataloader
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if model_type == "log_reg":
                outputs = model(images.reshape(images.shape[0],784))
                del images
            elif model_type == "cnn_mnist":
                outputs = model(images)
                del images
            elif model_type == "cnn_3c":
                outputs = model(images)
                del images
            elif model_type == "cnn_3c_small":
                outputs = model(images)
                del images
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    gc.collect()
    return 100*correct/total

def kmeans_sampling_log(dataset, unlab_idx, budget, device = None):
    """
    Selects points from kmeans clusters
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    budget: Points to sample
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    X = []
    for i in unlab_idx:
        inputs, labels = dataset[i]
        X.append(inputs.view(-1).cpu().numpy())
    X = np.array(X)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)  
    index = []
    d = np.hstack([kmeans.transform(X), np.array(unlab_idx).reshape(len(X),1)])
    for p in kmeans.labels_:
        dx = sorted(d[kmeans.labels_ == p], key = lambda x: x[p])
        index.extend([int(x[-1]) for x in dx[int(budget/10)]])
    return list(set(index))

def entropy_sampling(model, dataset, unlab_idx, budget, device = None, model_type = "log_reg"):
    """
    Selects points with highest entropy
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    budget: Points to sample
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda:1")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.parallel.DataParallel(model)
    model.eval()
    probs = []
    dataloader = dataset.get_data_label_loader(unlab_idx)
    if model_type == "log_reg":
        for i, (j,k) in enumerate(dataloader):
            if i == 0:
                probs = model(j.reshape(j.shape[0], 784).to(device)).detach().cpu()
                if probs.shape[-1] != 10 or len(probs.shape)<2:
                    probs = probs.reshape(int(probs.shape[-1]/10), 10)
                del j
            else:
                prob = model(j.reshape(j.shape[0], 784).to(device)).detach().cpu()
                if prob.shape[-1] != 10 or len(prob.shape)<2:
                    prob = prob.reshape(int(prob.shape[-1]/10), 10)
                probs = torch.cat((probs, prob), axis = 0)
                del j, prob
        probs = F.softmax(probs).detach().cpu().numpy()
    else:
        for i, (j,k) in enumerate(dataloader):
            if i == 0:
                probs = model(j.to(device)).detach().cpu()
                if probs.shape[-1] != 10 or len(probs.shape)<2:
                    probs = probs.reshape(int(probs.shape[-1]/10), 10)
                del j
            else:
                prob = model(j.to(device)).detach().cpu()
                if prob.shape[-1] != 10 or len(prob.shape)<2:
                    prob = prob.reshape(int(prob.shape[-1]/10), 10)
                probs = torch.cat((probs, prob), axis = 0)
                del j, prob
        probs = F.softmax(probs).detach().cpu().numpy()
    ent = entropy(probs.squeeze(), axis = 1)
    del probs
    R = list(zip(ent, unlab_idx))
    R.sort(reverse = True)
    if len(ent)<=budget:
      selection = [x[1] for x in R]
    else:
      selection = [x[1] for x in R[:budget]]
    gc.collect()
    return list(selection)


def engrad(model, dataset, unlab_idx, budget, num_clusters = 10, device = None, model_type = "log_reg"):
    """
    Selects points with diverse gradients when the gradients are computed on the entropy of the output of the extracted model w.r.t. x.
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    budget: Points to sample
    clusters: Clusters for kmeans | Default: 10
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda:1")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    
    dataloader = dataset.get_data_label_loader(unlab_idx)
    model.eval()
    
    if model_type == "log_reg":
        # op = model(data.reshape(data.shape[0],784))
        for i, (j,k) in enumerate(dataloader):
            j.requires_grad = True
            if i==0:
                op = model(j.reshape(j.shape[0], 784).to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = j.grad.reshape(j.shape[0], 784).detach().cpu().numpy()
                del op
            else:
                op = model(j.reshape(j.shape[0], 784).to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = np.concatenate((grad, j.grad.reshape(j.shape[0], 784).detach().cpu().numpy()))
                del op
    elif model_type == "cnn_mnist":
        for i, (j,k) in enumerate(dataloader):
            j.requires_grad = True
            if i==0:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = j.grad.reshape(j.shape[0], 784).detach().cpu().numpy()
                del op
            else:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = np.concatenate((grad, j.grad.reshape(j.shape[0], 784).detach().cpu().numpy()))
                del op
    elif model_type == "cnn_3c":
        for i, (j,k) in enumerate(dataloader):
            j.requires_grad = True
            if i==0:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = j.grad.reshape(j.shape[0], 150528).detach().cpu().numpy()
                del op
            else:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = np.concatenate((grad, j.grad.reshape(j.shape[0], 150528).detach().cpu().numpy()))
                del op
    elif model_type == "cnn_3c_small":
        for i, (j,k) in enumerate(dataloader):
            j.requires_grad = True
            if i==0:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = j.grad.reshape(j.shape[0], 3072).detach().cpu().numpy()
                del op
            else:
                op = model(j.to(device))
                loss = torch.sum(entr(F.softmax(op, dim = 0)))
                loss.backward()
                grad = np.concatenate((grad, j.grad.reshape(j.shape[0], 3072).detach().cpu().numpy()))
                del op
    del loss
    X = grad
    del grad
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)  
    d = np.hstack([kmeans.transform(X), np.array(unlab_idx).reshape(len(X),1)])
    del X
    indices = np.zeros((num_clusters, len(d)), int)
    for i in range(len(indices)):
        indices[i] = np.array([x[-1] for x in sorted(d, key = lambda x: x[i])])

    t = int(budget/num_clusters)
    while len(np.unique(indices[:,:t]))<budget and t<len(unlab_idx):
        t += 1
    index = list(np.unique(indices[:,:t].flatten()))
    gc.collect()
    return index


def loss_dep(model, dataset, unlab_idx, prev_idx, budget, num_clusters = 10, device = None, model_type = "log_reg"):
    """
    Selects points from the already queried set with highest loss. Finds new points closer to these highest loss points
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    prev_idx: Indices of the 'dataset' that have already been used for query
    budget: Points to sample
    clusters: Clusters for kmeans | Default: 10
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
            print("Using CPU")
            device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda:1")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    model.eval()
    X = []
    y = []
    prev_loader = dataset.get_data_label_loader(prev_idx)
    y = dataset.get_label(prev_idx)

    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    j = 0
    if model_type == "log_reg":
        for i,(j,k) in enumerate(prev_loader):
            if i == 0:
                y_hat = model(j.reshape(j.shape[0],784).to(device)).detach()
                if y_hat.shape[-1] != 10 or len(y_hat.shape)<2:
                    y_hat = y_hat.reshape(int(y_hat.shape[-1]/10),10)
                del j
            else:
                out = model(j.reshape(j.shape[0],784).to(device)).detach()
                if out.shape[-1] != 10 or len(out.shape)<2:
                    out = out.reshape(int(out.shape[-1]/10),10)
                y_hat = torch.cat((y_hat,out))
                del j
        y_hat = torch.tensor(y_hat)
    elif model_type == "cnn_mnist":
        for i,(j,k) in enumerate(prev_loader):
            if i == 0:
                y_hat = model(j.to(device)).detach()
                if y_hat.shape[-1] != 10 or len(y_hat.shape)<2:
                    y_hat = y_hat.reshape(int(y_hat.shape[-1]/10),10)
                del j
            else:
                out = model(j.to(device)).detach()
                if out.shape[-1] != 10 or len(out.shape)<2:
                    out = out.reshape(int(out.shape[-1]/10),10)
                y_hat = torch.cat((y_hat,out))
                del j
        y_hat = torch.tensor(y_hat)
    elif model_type == "cnn_3c":
        for i,(j,k) in enumerate(prev_loader):
            if i == 0:
                y_hat = model(j.to(device)).detach()
                if y_hat.shape[-1] != 10 or len(y_hat.shape)<2:
                    y_hat = y_hat.reshape(int(y_hat.shape[-1]/10),10)
                del j
            else:
                out = model(j.to(device)).detach()
                if out.shape[-1] != 10 or len(out.shape)<2:
                    out = out.reshape(int(out.shape[-1]/10),10)
                y_hat = torch.cat((y_hat, out), axis = 0)
                del j
        y_hat = torch.tensor(y_hat)
    elif model_type == "cnn_3c_small":
        for i,(j,k) in enumerate(prev_loader):
            if i == 0:
                y_hat = model(j.to(device)).detach()
                if y_hat.shape[-1] != 10 or len(y_hat.shape)<2:
                    y_hat = y_hat.reshape(int(y_hat.shape[-1]/10),10)
                    del j
            else:
                out = model(j.to(device)).detach()
                if out.shape[-1] != 10 or len(out.shape)<2:
                    out = out.reshape(int(out.shape[-1]/10),10)
                y_hat = torch.cat((y_hat, out), axis = 0)
                del j
        y_hat = torch.tensor(y_hat)
    loss = criterion(y_hat.cpu(),y.cpu()).detach()
    del y, y_hat
    loss_id = torch.argsort(loss.cpu(), descending = True)
    if model_type == "cnn_3c":
        X_comp = torch.empty((num_clusters, 150528)).to("cpu")
    elif model_type == "cnn_3c_small":
        X_comp = torch.empty((num_clusters, 3072)).to("cpu")
    elif model_type in ["log_reg", "cnn_mnist"]:
        X_comp = torch.empty((num_clusters, 784)).to("cpu")
    if model_type == "cnn_3c":
        for i in range(num_clusters):
            X_comp[i] = dataset.get_data(prev_idx)[loss_id[i]].reshape(150528).to("cpu")
    elif model_type == "cnn_3c_small":
        for i in range(num_clusters):
            X_comp[i] = dataset.get_data(prev_idx)[loss_id[i]].reshape(3072).to("cpu")
    elif model_type in ["log_reg", "cnn_mnist"]:
        for i in range(num_clusters):
            X_comp[i] = dataset.get_data(prev_idx)[loss_id[i]].reshape(784).to("cpu")
    del loss_id
    norms = torch.empty((len(X_comp), len(unlab_idx)))

    for i in range(len(X_comp)):
        if model_type == "cnn_3c":
            diff = X_comp[i, :] - dataset.get_data(unlab_idx).reshape(len(unlab_idx), 150528)
            norms[i, :] = torch.linalg.norm(diff, axis=1)
            del diff
        elif model_type == "cnn_3c_small":
            diff = X_comp[i, :] - dataset.get_data(unlab_idx).reshape(len(unlab_idx), 3072)
            norms[i, :] = torch.linalg.norm(diff, axis=1)
            del diff
        elif model_type in ["log_reg", "cnn_mnist"]:
            diff = X_comp[i, :].cpu() - dataset.get_data(unlab_idx).reshape(len(unlab_idx), 784).cpu()
            norms[i, :] = torch.linalg.norm(diff, axis=1)
            del diff
    
    del X_comp
    sorted_indices = torch.argsort(norms, dim=1)
    del norms

    t = int(budget/num_clusters)
    while len(np.unique(sorted_indices[:,:t]))<budget and t<len(unlab_idx):
        t += 1
    idxs = np.unique(sorted_indices[:,:t])
    del sorted_indices
    index2 = list(np.array(unlab_idx)[list(idxs)])
    gc.collect()
    return list(set(index2))

def kmeans_train(model, dataset, val_loader, test_loader, sample_list, epochs, LR = 0.001, bs = 20, model_name = "entropy_log.pt", device = None):
    """
    model: The extracted model that is to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    val_loader: Validation dataloader (pytorch dataloader)
    test_loader: Test dataloader (pytorch dataloader)
    sample_list: list of samples to be used for model extraction in every round
    epochs: Epochs per round
    LR: Learning rate | Default: 0.001
    model_name: Name of the model to be saved. User may specify path here. | Default: "entropy_log.pt"
    device: Any one from ["cuda", "cpu", "mps"] | Default: None -> "cpu"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    all_idx = range(len(dataset))
    train_idx = random.sample(range(0,len(dataset)), sample_list[0])
    train_data = Subset(dataset, train_idx)
    remain_idx = list(set(all_idx) - set(train_idx))
    test_acc_list = [test(model, test_loader, device = device)]
    print("Test accuracy: ", test_acc_list[-1])
    train_loss_list = []
    valid_loss_list = []
    samples = [len(train_idx)]
    print("Training samples: ", samples[-1])
    train_list, val_list = train(train_data, val_loader, model, epochs = epochs, criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True)
    del train_data
    train_loss_list.extend(train_list)
    valid_loss_list.extend(val_list)
    test_acc_list.append(test(model, test_loader, device = device))
    print("Test accuracy: ", test_acc_list[-1])

    for sample in sample_list[1:]:
        print(sample)
        new_idx = kmeans_sampling_log(dataset, remain_idx, budget = sample - len(train_idx), device = "cuda")
        remain_idx = list(set(remain_idx) - set(new_idx))
        train_idx = list(set(train_idx + new_idx))
        train_data = Subset(dataset, train_idx)
        train_list, val_list = train(train_data, val_loader, model, epochs = int(epochs), criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True, val_loss_list = valid_loss_list)
        del train_data
        valid_loss_list.extend(val_list)
        print("Testing")
        test_acc_list.append(test(model, test_loader, device = device))
        print("Test accuracy: ", test_acc_list[-1])

    return train_loss_list, valid_loss_list, test_acc_list

def entropy_train(model, dataset, val_loader, test_loader, sample_list, epochs, LR = 0.001, model_name = "entropy_log.pt", device = None, model_type = "log_reg"):
    """
    model: The extracted model that is to be trained
    dataset: The public dataset
    val_loader: Validation dataloader (pytorch dataloader)
    test_loader: Test dataloader (pytorch dataloader)
    sample_list: list of samples to be used for model extraction in every round
    epochs: Epochs per round
    LR: Learning rate | Default: 0.001
    model_name: Name of the model to be saved. User may specify path here. | Default: "entropy_log.pt"
    device: Any one from ["cuda", "cpu", "mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")
    model.to(device)
    all_idx = range(len(dataset))
    train_idx = random.sample(range(0,len(dataset)), sample_list[0])
    train_data = Subset(dataset, train_idx)
    remain_idx = list(set(all_idx) - set(train_idx))
    test_acc_list = [test(model, test_loader, device = device)]
    print("Test accuracy: ", test_acc_list[-1])
    train_loss_list = []
    valid_loss_list = []
    samples = [len(train_idx)]
    print("Training samples: ", samples[-1])
    train_list, val_list = train(train_data, val_loader, model, epochs = epochs, criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True)
    del train_data
    train_loss_list.extend(train_list)
    # valid_loss_list = val_list
    valid_loss_list.extend(val_list)
    test_acc_list.append(test(model, test_loader, device = device))
    print("Test accuracy: ", test_acc_list[-1])

    for sample in sample_list[1:]:
        new_idx = entropy_sampling(model, dataset, remain_idx, budget = sample - len(train_idx), device = device, model_type = model_type)
        remain_idx = list(set(remain_idx) - set(new_idx))
        train_idx = list(set(train_idx + new_idx))
        train_data = Subset(dataset, train_idx)
        train_list, val_list = train(train_data, val_loader, model, epochs = int(epochs), criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True, val_loss_list = valid_loss_list)
        del train_data
        valid_loss_list.extend(val_list)
        print("Testing")
        test_acc_list.append(test(model, test_loader, device = device))
        print("Test accuracy: ", test_acc_list[-1])

    return train_loss_list, valid_loss_list, test_acc_list

def marich(model, dataset, val_loader, test_loader, budget = 300, init_points = 1000, rounds = 20, epochs = 20, LR = 0.001, gamma1 = 0.8, gamma2 = 0.8, model_name = "extracted.pt", sampling = "all_elg", device = None, model_type = "log_reg"):
    """
    model: The extracted model that is to be trained
    dataset: The public dataset
    val_loader: Validation dataloader (pytorch dataloader)
    test_loader: Test dataloader (pytorch dataloader)
    budget: Initial budget | Default: 300
    init_points: Randomly selected points for initial training of model | Default: 1000
    rounds: Rounds of samplings | Default: 20
    epochs: Epochs per round | Default: 20
    LR: Learning rate | Default: 0.001
    gamma1: fraction of points to be chosen after first sampling step | Default: 0.8
    gamma2: fraction of points to be chosen after second sampling step | Default: 0.8
    model_name: Name of the model to be saved. User may specify path here. | Default: "extracted.pt"
    sampling: Any of ["all_elg", "all_egl", "entropy", "loss", "engrad", "entropy-loss", "entropy-engrad", "loss-engrad", "random"] | Default: "all"
    device: Any one from ["cuda", "cpu", "mps"] | Default: None -> "cpu"
    model_type: Kind of model to be used for extraction | Any one from ["log_reg", "cnn_mnist", "cnn_cifar"] | Default: "log_reg"
    """
    if device == None:
        print("Using CPU")
        device = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device("cuda")
        else:
            print("Cuda not found. Using CPU.")
            device =torch.device("cpu")
    elif device == "mps":
        if torch.has_mps:
            print("Using MPS")
            device = torch.device("mps")
        else:
            print("MPS not found. Using CPU")
            device = torch.device("cpu")

    model.to(device)
    all_idx = range(len(dataset))
    train_idx = random.sample(range(0,len(dataset)), init_points)
    train_data = dataset.get_dataset(train_idx)
    remain_idx = list(set(all_idx) - set(train_idx))
    test_acc_list = [test(model, test_loader, device = device, model_type = model_type)]
    samples = [0]
    print("Test accuracy: ", test_acc_list[-1])
    train_loss_list = []
    valid_loss_list = []
    samples.append(len(train_idx))
    print("Training samples: ", samples[-1])
    train_list, val_list = train(train_data, val_loader, model, epochs = epochs, criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True, model_type = model_type)
    del train_data
    train_loss_list.extend(train_list)
    valid_loss_list.extend(val_list)
    test_acc_list.append(test(model, test_loader, device = device, model_type = model_type))
    print("Test accuracy: ", test_acc_list[-1])
  
    for r in range(rounds):
        print("Round: ", r+1)
        budget = budget*(1.01)

        # Sampling using entropy
        new_idx = remain_idx
        if sampling == 'all_elg':
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/(gamma1*gamma2)), device = device, model_type = model_type)

            # Sampling using loss
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma1*len(new_idx)), device = device, model_type = model_type)

            # gradient sampling
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma2*len(new_idx)), device = device, num_clusters = 10, model_type = model_type)

        elif sampling == 'all_egl':
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/(gamma1*gamma2)), device = device, model_type = model_type)

            # gradient sampling
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 10, model_type = model_type)
            
            # Sampling using loss
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma2*len(new_idx)), device = device, model_type = model_type)
        
        elif sampling == "entropy":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget), device = device, model_type = model_type)
        
        elif sampling == "loss":
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = int(budget), device = device, model_type = model_type)
        
        elif sampling == "engrad":
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(budget), device = device, num_clusters = 10, model_type = model_type)
        
        elif sampling == "entropy-loss":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/gamma1), device = device, model_type = model_type)
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma1*len(new_idx)), device = device, model_type = model_type)
        
        elif sampling == "entropy-engrad":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/gamma1), device = device, model_type = model_type)
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 10, model_type = model_type)
        
        elif sampling == "loss-engrad":
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = int(budget/gamma1), device = device, model_type = model_type)
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 10, model_type = model_type)
        
        elif sampling == 'random':
            print("Using random sampling strategy on ", len(new_idx),"points")
            new_idx = random.sample(new_idx, int(budget))


        samples.append(len(train_idx)+len(new_idx))

        remain_idx = list(set(remain_idx) - set(new_idx))

        train_idx = list(set(train_idx + new_idx))

        train_data = dataset.get_dataset(train_idx)

        print("Budget: ", samples[-1])
        print("Training samples: ", len(train_idx))
        print("Training")
        
        epochs = 1.02*epochs

        train_list, val_list = train(train_data, val_loader, model, epochs = int(epochs), criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True, val_loss_list = valid_loss_list, model_type = model_type)
        del train_data
        train_loss_list.extend(train_list)

        torch.save(model, "./extracted_models/train_step/"+model_name)

        valid_loss_list.extend(val_list)

        print("Testing")
        test_acc_list.append(test(model, test_loader, device = device, model_type = model_type))
        print("Test accuracy: ", test_acc_list[-1])
        gc.collect()

    return train_loss_list, valid_loss_list, test_acc_list, samples
