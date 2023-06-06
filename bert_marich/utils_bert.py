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
from transformers import get_linear_schedule_with_warmup
import random
from tqdm import tqdm
import datetime
from transformers import BertTokenizer

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

    def get_data_label_loader(self, idx, batch_size = 8):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        y = torch.tensor(self.Y)[idx]
        
        return_set = basic_dataset(x,y)
        return torch.utils.data.DataLoader(return_set, batch_size = batch_size)
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(traindata, valloader, model, epochs, criterion, lr = 0.001, device = None, model_name = "logreg.pt", save = True, val_loss_list = []):
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
    trainloader = DataLoader(traindata, batch_size = 8, shuffle = True)
    train_loss_list = []
    optimizer = optim.AdamW(model.parameters(),
                  lr = lr, 
                  eps = 1e-8 
                )
    total_steps = len(trainloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    train_loss_list = []
    val_loss_list = []
    for i in range(epochs):
        print("Epoch: ",i+1)
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(trainloader)):
            batch,labels = batch
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(trainloader), elapsed))

            model.zero_grad()        
            outputs = model(batch["input_ids"].squeeze(dim = 1).to(device),
                            batch["attention_mask"].squeeze(dim = 1).to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            del batch, labels, loss
            torch.cuda.empty_cache()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(trainloader)            
        
        # Store the loss value for plotting the learning curve.
        train_loss_list.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        t0 = time.time()
        model.eval()
        print("Running Validation")
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in tqdm(valloader):
            
            batch,labels = batch
            with torch.no_grad():        
                outputs = model(batch["input_ids"].squeeze(dim = 1).to(device), 
                            batch["attention_mask"].squeeze(dim = 1).to(device))
            logits = outputs
            loss = criterion(outputs, labels.to(device))
            eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
            del batch, labels
        avg_val_loss = eval_loss / len(valloader)
        val_loss_list.append(avg_val_loss)
        if save:    
            if val_loss_list[-1]<=min(val_loss_list):
                print("validation loss minimum, saving model")
                torch.save(model.state_dict(), "./extracted_models/"+model_name)
        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        
        gc.collect()
    return train_loss_list, val_loss_list


def test(model, testloader, device = None):
    """
    model: Model to be tested
    testedloader: Pytorch dataloader
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
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
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in testloader:
        
            batch,labels = batch
            with torch.no_grad():        
                outputs = model(batch["input_ids"].squeeze(dim = 1).to(device), 
                            batch["attention_mask"].squeeze(dim = 1).to(device))
            logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
            del batch, labels

    gc.collect()
    return 100*eval_accuracy/nb_eval_steps

def entropy_sampling(model, dataset, unlab_idx, budget, device = None):
    """
    Selects points with highest entropy
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
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model)
    model.eval()
    probs = []
    dataloader = dataset.get_data_label_loader(unlab_idx)


    for i, batch in enumerate(dataloader):
        batch,labels = batch
        if i == 0:
            probs = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device)).detach().cpu()
            if probs.shape[-1] != 5 or len(probs.shape)<2:
                probs = probs.reshape(int(probs.shape[-1]/5), 5)
            del batch
        else:
            prob = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device)).detach().cpu()
            if prob.shape[-1] != 5 or len(prob.shape)<2:
                prob = prob.reshape(int(prob.shape[-1]/5), 5)
            probs = torch.cat((probs, prob), axis = 0)
            del batch, prob
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


def engrad(model, dataset, unlab_idx, budget, num_clusters = 5, device = None):
    """
    Selects points with diverse gradients when the gradients are computed on the entropy of the output of the extracted model w.r.t. x.
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    budget: Points to sample
    clusters: Clusters for kmeans | Default: 10
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
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
    
    
    for i, batch in enumerate(dataloader):
        batch,labels = batch
        if i==0:
            op = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device))
            loss = torch.sum(entr(F.softmax(op, dim = 0)))
            loss.backward()
            grad = model.bert.embeddings.word_embeddings.weight.grad[batch["input_ids"]].detach().cpu().numpy()
            grad = grad.reshape(grad.shape[0], 512*768)
            del op
        else:
            op = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device))
            loss = torch.sum(entr(F.softmax(op, dim = 0)))
            loss.backward()
            g = model.bert.embeddings.word_embeddings.weight.grad[batch["input_ids"]].detach().cpu().numpy()
            g = g.reshape(g.shape[0], 512*768)
            grad = np.concatenate((grad, g))
            del op, g
    
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


def loss_dep(model, dataset, unlab_idx, prev_idx, budget, num_clusters = 10, device = None):
    """
    Selects points from the already queried set with highest loss. Finds new points closer to these highest loss points
    model: Extracted model to be trained
    dataset: The public dataset whose labels are the outputs of the target model
    unlab_idx: Indices of the 'dataset' that are yet to be used
    prev_idx: Indices of the 'dataset' that have already been used for query
    budget: Points to sample
    clusters: Clusters for kmeans | Default: 10
    device: Any of ["cuda", "cpu","mps"] | Default: None -> "cpu"
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
    for i,batch in enumerate(prev_loader):
        batch, labels = batch
        if i == 0:
            y_hat = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device)).detach()
            if y_hat.shape[-1] != 5 or len(y_hat.shape)<2:
                y_hat = y_hat.reshape(int(y_hat.shape[-1]/5),5)
            del batch
        else:
            out = model(batch["input_ids"].squeeze(dim = 1).to(device), batch["attention_mask"].squeeze(dim = 1).to(device)).detach()
            if out.shape[-1] != 5 or len(out.shape)<2:
                out = out.reshape(int(out.shape[-1]/5),5)
            y_hat = torch.cat((y_hat,out))
            del batch
    y_hat = torch.tensor(y_hat)
    
    loss = criterion(y_hat.cpu(),y.cpu()).detach()
    del y, y_hat
    loss_id = torch.argsort(loss.cpu(), descending = True)
    X_comp = torch.empty((num_clusters, 512)).to("cpu")

    for i in range(num_clusters):
        X_comp[i] = dataset.get_data(prev_idx)[loss_id[i]].reshape(512).to("cpu")
    del loss_id
    norms = torch.empty((len(X_comp), len(unlab_idx)))

    for i in range(len(X_comp)):
        diff = X_comp[i, :] - dataset.get_data(unlab_idx).reshape(len(unlab_idx), 512)
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

def marich(model, dataset, val_loader, test_loader, budget = 300, init_points = 1000, rounds = 20, epochs = 20, LR = 0.001, gamma1 = 0.8, gamma2 = 0.8, model_name = "extracted.pt", sampling = "all_elg", device = None):
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
    device_samp = torch.device("cuda:1")
    all_idx = range(len(dataset))
    train_idx = random.sample(range(0,len(dataset)), init_points)
    train_data = dataset.get_dataset(train_idx)

    remain_idx = list(set(all_idx) - set(train_idx))
    test_acc_list = [test(model, test_loader, device = device)]
    samples = [0]
    print("Test accuracy: ", test_acc_list[-1])
    train_loss_list = []
    valid_loss_list = []
    samples.append(len(train_idx))
    print("Training samples: ", samples[-1])
    train_list, val_list = train(train_data, val_loader, model, epochs = epochs, criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True)
    del train_data
    train_loss_list.extend(train_list)
    valid_loss_list.extend(val_list)
    test_acc_list.append(test(model, test_loader, device = device))
    print("Test accuracy: ", test_acc_list[-1])
  
    for r in range(rounds):
        print("Round: ", r+1)
        budget = budget*(1.01)


        # Sampling using entropy
        new_idx = remain_idx
        if sampling == 'all_elg':
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/(gamma1*gamma2)), device = device)

            # Sampling using loss
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma1*len(new_idx)), device = device)

            # gradient sampling
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma2*len(new_idx)), device = device, num_clusters = 5)

        elif sampling == 'all_egl':
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/(gamma1*gamma2)), device = device)

            # gradient sampling
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 5)
            
            # Sampling using loss
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma2*len(new_idx)), device = device)
        
        elif sampling == "entropy":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget), device = device)
        
        elif sampling == "loss":
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = int(budget), device = device)
        
        elif sampling == "engrad":
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(budget), device = device, num_clusters = 5)
        
        elif sampling == "entropy-loss":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/gamma1), device = device)
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = (gamma1*len(new_idx)), device = device)
        
        elif sampling == "entropy-engrad":
            print("Using entropy sampling on ", len(new_idx)," points")
            new_idx = entropy_sampling(model, dataset, remain_idx, budget = int(budget/gamma1), device = device)
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 5)
        
        elif sampling == "loss-engrad":
            print("Using loss sampling on ", len(new_idx)," points")
            new_idx = loss_dep(model, dataset, new_idx, train_idx, budget = int(budget/gamma1), device = device)
            print("Using gradient sampling on ", len(new_idx), " points")
            new_idx = engrad(model, dataset, new_idx, budget = int(gamma1*len(new_idx)), device = device, num_clusters = 5)
        
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
        train_list, val_list = train(train_data, val_loader, model, epochs = int(epochs), criterion = torch.nn.CrossEntropyLoss(), lr = LR, device = device, model_name = model_name, save =True, val_loss_list = valid_loss_list)
        del train_data
        train_loss_list.extend(train_list)
        torch.save(model, "./extracted_models/train_step/"+model_name)
        valid_loss_list.extend(val_list)

        print("Testing")
        test_acc_list.append(test(model, test_loader, device = device))
        print("Test accuracy: ", test_acc_list[-1])
        gc.collect()

    return train_loss_list, valid_loss_list, test_acc_list, samples
