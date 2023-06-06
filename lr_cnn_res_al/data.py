import numpy as np
import torch
from torchvision import datasets, transforms
from nets import LogisticRegression, ResNet, CNN, CNN_img
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class ImageNet_dataset(Dataset):
    def __init__(self, X, transform = None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        
        if self.transform:
            x = self.X[idx]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        else:
            x = self.X[idx]
        return x
    
class img_dataset(Dataset):
    def __init__(self, X, transform = None):
        self.X = X
        # self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        return x

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
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

def get_EMNIST_log(handler):
    with open('./imagenet_target/emnist_lr_x.pkl', 'rb') as f:
        unlab_x = pickle.load(f)

    with open('./imagenet_target/emnist_lr_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                    ])
    data_test = datasets.MNIST('../mnist/train/', download=True, train=False, transform=transform)
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)
    # return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_CIFAR10_log(handler):
    with open('./imagenet_target/cifar_lr_x.pkl', 'rb') as f:
        unlab_x = pickle.load(f)

    with open('./imagenet_target/cifar_lr_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                    ])
    data_test = datasets.MNIST('../mnist/train/', download=True, train=False, transform=transform)
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)

def get_EMNIST_cnn(handler):
    with open('./imagenet_target/emnist_cnn_x.pkl', 'rb') as f:
        unlab_x = pickle.load(f)

    with open('./imagenet_target/emnist_cnn_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    data_test = datasets.MNIST('../mnist/train/', download=True, train=False, transform=transform)
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)
    

def get_CIFAR10_cnn(handler):
    with open('./imagenet_target/cifar_cnn_x.pkl', 'rb') as f:
        unlab_x = pickle.load(f)

    with open('./imagenet_target/cifar_cnn_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    data_test = datasets.MNIST('../mnist/train/', download=True, train=False, transform=transform)
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)

def get_Imagenet_res_cnn(handler):
    imagenet = unpickle("val_data")
    imagenet = torch.tensor(np.float32(imagenet['data'].reshape(50000,3,32,32)/255))
    train_transform = transforms.Compose([
        transforms.Normalize([0.473, 0.450, 0.401], [0.258, 0.251, 0.265])
        ])
    unlab_x = img_dataset(imagenet, transform = train_transform)
    with open('./imagenet_target/imagnet_resnet_small_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    data_test = datasets.CIFAR10('./cifar10/train/', download=True, train=False, transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ]))
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)

def get_Imagenet_res_res18(handler):
    unlab_x = torch.load("./imagenet_target/imagenet_resnet.pt")
    with open('./imagenet_target/imagnet_resnet_small_y.pkl', 'rb') as f:
        unlab_y = pickle.load(f)
    data_test = datasets.CIFAR10('./cifar10/train/', download=True, train=False, transform = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                        ]))
    Y_test = torch.stack([data_test[i][0] for i in range(len(data_test))])
    return Data(unlab_x, unlab_y, Y_test, torch.LongTensor(data_test.targets), handler)
    

def get_STL10(handler):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    resnet = ResNet().to(device)
    if torch.cuda.is_available():
        resnet.load_state_dict(torch.load("./targets/resnet18_cifar.pt"))
    else:
        resnet.load_state_dict(torch.load("./targets/resnet18_cifar.pt", map_location=torch.device('cpu')))
    data_train = datasets.STL10(root="./data/STL10", split="train", download=True, transform= transforms.Compose([
                                                                                                transforms.Resize(32),
                                                                                                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261))]))
    data_train2 = datasets.STL10(root="./data/STL10", split="test",download=True, transform = transforms.Compose([
                                                                                                transforms.Resize(32),
                                                                                                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261))]))
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform = transforms.Compose([
                                                                                                transforms.Resize(224),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))


                                                                                                # [transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
                                                                                                # transforms.Resize(224),
                                                                                                # transforms.ToTensor(),
                                                                                                # transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))]))


                                                                                                # transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
                                                                                                # transforms.Resize(224),
                                                                                                # transforms.ToTensor(),
                                                                                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    attack_set = ConcatDataset([data_train,data_train2])
    unlab_x = []
    unlab_y = []
    print("Getting target(x)")
    path = Path("./stl10_y.npy")
    if path.is_file():
        unlab_x = np.load("./stl10_x.npy", allow_pickle = True)
        unlab_y = np.load("./stl10_y.npy", allow_pickle = True)
    else:
        for j,k in tqdm(attack_set):
            unlab_y.append(torch.argmax(resnet(j.to(device).unsqueeze(dim = 0))[0], dim = 1)[0].item())
            unlab_x.append(j.detach().cpu())
            del j
        unlab_x = np.array(unlab_x)
        # unlab_y = torch.LongTensor(unlab_y)
        np.save("./stl10_x.npy", unlab_x)
        np.save("./stl10_y.npy", np.array(unlab_y))
    return Data(unlab_x[:40000], torch.LongTensor(unlab_y)[:40000], torch.stack([x[0] for x in data_test])[:40000], torch.LongTensor(data_test.targets)[:40000], handler)


def get_ImageNet(handler):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    resnet = ResNet().to(device)
    if torch.cuda.is_available():
        resnet.load_state_dict(torch.load("./targets/my_resnet.pt"))
    else:
        resnet.load_state_dict(torch.load("./targets/my_resnet.pt", map_location=torch.device('cpu')))
    path = Path("./imagenet_y.npy")
    if path.is_file() == False:
        imagenet = unpickle("./Imagenet32_val/val_data")['data'].reshape(50000,3,32,32)
        imgnet_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([120.608,114.791,102.286],[65.763,64.028,67.674])])
        data_train = ImageNet_dataset(imagenet, transform = imgnet_transform)
    data_train = datasets.CIFAR100('./data/CIFAR100', train=False, download=True, transform = transforms.Compose([
                                                                                                transforms.ToTensor()]))
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform = transforms.Compose([
                                                                                                transforms.ToTensor()]))
                                                                                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    unlab_x = []
    unlab_y = []
    print("Getting target(x)")
    
    if path.is_file() == False:
        unlab_x = np.load("./imagenet_x.npy", allow_pickle = True)
        unlab_y = np.load("./imagenet_y.npy", allow_pickle = True)
    else:
        for j,k in tqdm(data_train):
            j = j.to(device)
            unlab_y.append(torch.argmax(resnet(j.to(device).unsqueeze(dim = 0))[0], dim = 1)[0].item())
            unlab_x.append(j.detach().cpu())
        unlab_x = np.array(unlab_x)
        # unlab_y = torch.LongTensor(unlab_y)
        np.save("./cifar100_x.npy", unlab_x)
        np.save("./cifar100_y.npy", np.array(unlab_y))
    return Data(unlab_x, torch.LongTensor(unlab_y), torch.stack([x[0] for x in data_test])[:40000], torch.LongTensor(data_test.targets)[:40000], handler)


def get_IMGNET_cnn(handler):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    cnn = CNN_img().to(device)
    if torch.cuda.is_available():
        cnn.load_state_dict(torch.load("./targets/cnn_cifar.pt"))
    else:
        cnn.load_state_dict(torch.load("./targets/cnn_cifar.pt", map_location=torch.device('cpu')))
    cifar_training_transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    path = Path("./imagenet_y.npy")
    if path.is_file() == False:
        imagenet = unpickle("val_data")
        imagenet = imagenet['data'].reshape(50000,3,32,32)/255
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([np.mean(imagenet[:,i,:,:]) for i in range(3)], [np.std(imagenet[:,i,:,:]) for i in range(3)])
        ])

    data_test = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform = cifar_training_transform)
    unlab_x = []
    unlab_y = []
    print("Getting target(x)")
    path = Path("./imagenet_y.npy")
    if path.is_file():
        print("Loading existing target(x)")
        unlab_x = np.load("./imagenet_x.npy", allow_pickle = True)
        unlab_y = np.load("./imagenet_y.npy", allow_pickle = True)
    else:
        for j in tqdm(imagenet):
            j = torch.tensor(j, dtype = torch.float).to(device)
            unlab_y.append(torch.argmax(cnn(j.unsqueeze(dim = 0))[0]).detach().cpu())
            unlab_x.append(j.detach().cpu())
            del j
        unlab_x = np.array(unlab_x)
        np.save("./imagenet_x.npy", unlab_x)
        np.save("./imagenet_y.npy", np.array(unlab_y))
    return Data(unlab_x, torch.LongTensor(unlab_y), torch.stack([x[0] for x in data_test])[:40000], torch.LongTensor(data_test.targets)[:40000], handler)