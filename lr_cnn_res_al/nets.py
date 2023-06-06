import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torchvision import models
from scipy.stats import entropy
from torchvision import datasets, transforms
from handlers import ImageNet_Handler


transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                            ])
data2 = datasets.CIFAR10(root='./cifar10/train/', train=False, download=True, transform=transform)
cifar_x = torch.stack([data2[i][0] for i in range(len(data2))])
data2 = ImageNet_Handler(cifar_x, torch.LongTensor(data2.targets))
loader2 = DataLoader(data2, shuffle=False, batch_size = 64, num_workers = 1)


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
        if self.params["name"] == "EMNIST_log":
            optimizer = optim.AdamW(self.clf.parameters(), lr=0.02, weight_decay = 0.001)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.02, epochs=n_epoch, steps_per_epoch=len(loader))
        elif self.params["name"] == "CIFAR10_log":
            optimizer = optim.AdamW(self.clf.parameters(), lr=0.02, weight_decay = 0.001)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.02, epochs=n_epoch, steps_per_epoch=len(loader))
        elif self.params["name"] == "EMNIST_cnn":
            optimizer = optim.SGD(self.clf.parameters(), lr=0.015, weight_decay = 0.001)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.95)
        elif self.params["name"] == "CIFAR10_cnn":
            optimizer = optim.SGD(self.clf.parameters(), lr=0.03, weight_decay = 0.001)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.95)
        elif self.params["name"] == "Imagenet_res_cnn":
            optimizer = optim.SGD(self.clf.parameters(), lr=0.2, weight_decay = 0.0001)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.85)
        elif self.params["name"] == "Imagenet_res_res18":
            config = {                                                                                                                                                                                                          
            'batch_size': 16,
            'lr': 8.0505e-05,
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
            } 
            optimizer = optim.Adam(
            self.clf.parameters(), 
            lr=config['lr'], 
            betas=(config['beta1'], config['beta2']), 
            amsgrad=config['amsgrad'], 
            )
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        losses = []
        val_losses = []
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            loss_e = 0
            tr_acc = 0
            total = 0
            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                self.clf.train()
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                cost = nn.CrossEntropyLoss()
                loss = cost(out, y)
                tr_acc += torch.sum(torch.argmax(out, dim = 1) == y)
                total += len(x)
                del x
                del out
                del y
                del e1
                loss.backward()
                optimizer.step()
                if self.params["net"] in ["ResNet", "LogReg"]:
                    lr_scheduler.step()
                loss_e += float(loss)
            del loss
            losses.append(loss_e/len(loader))
            tr_acc = tr_acc/total
            print("Train accuracy = ", tr_acc)

            loss_v = 0
            for batch_idx, (x, y, idxs) in enumerate(val_loader):
                x,y = x.to(self.device), y.to(self.device)
                self.clf.eval()
                out, e1 = self.clf(x)
                cost = nn.CrossEntropyLoss()
                loss_val = float(cost(out, y))
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
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        target_preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        if self.params["data"] in ["EMNIST_log", "CIFAR10_log"]:
            if torch.cuda.is_available():
                model = LogisticRegression().to(self.device)
                model.load_state_dict(torch.load("./targets/logreg_mnist.pt"))
            else:
                model = LogisticRegression().cpu()
                model.load_state_dict(torch.load("./targets/logreg_mnist.pt", map_location=torch.device('cpu')))
        if self.params["data"] in ["EMNIST_cnn", "CIFAR10_cnn"]:
            if torch.cuda.is_available():
                model = CNN().to(self.device)
                model.load_state_dict(torch.load("./targets/cnn_mnist.pt"))
            else:
                model = CNN().cpu()
                model.load_state_dict(torch.load("./targets/cnn_mnist.pt", map_location=torch.device('cpu')))
        if self.params["data"] == "Imagenet_res_cnn":
            if torch.cuda.is_available():
                model = ResNet_small().to(self.device)
                model.load_state_dict(torch.load("./targets/small_resnet.pt"))
            else:
                model = ResNet_small().cpu()
                model.load_state_dict(torch.load("./targets/small_resnet.pt", map_location=torch.device('cpu')))
        if self.params["data"] == "Imagenet_res_res18":
            if torch.cuda.is_available():
                model = ResNet_small().to(self.device)
                model.load_state_dict(torch.load("./targets/small_resnet.pt"))
            else:
                model = ResNet_small().cpu()
                model.load_state_dict(torch.load("./targets/small_resnet.pt", map_location=torch.device('cpu')))
        model.eval()
        
        if self.params["data"] not in ["Imagenet_res_cnn", "Imagenet_res_res18"]:
            with torch.no_grad():
                for x, y, idxs in tqdm(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x.float())
                    target_out, _ = model(x.float())
                    pred = out.max(1)[1]
                    target_pred = target_out.max(1)[1]
                    preds[idxs] = pred.cpu()
                    target_preds[idxs] = target_pred.cpu()
        else:
            with torch.no_grad():
                for x, y, idxs in tqdm(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x.float())
                    # target_out, _ = model(x)
                    pred = out.max(1)[1]
                    # target_pred = target_out.max(1)[1]
                    preds[idxs] = pred.cpu()
                    # target_preds[idxs] = target_pred.cpu()
                for x, y, idxs in tqdm(loader2):
                    x, y = x.to(self.device), y.to(self.device)
                    # out, e1 = self.clf(x)
                    target_out, _ = model(x.float())
                    # pred = out.max(1)[1]
                    target_pred = target_out.max(1)[1]
                    # preds[idxs] = pred.cpu()
                    target_preds[idxs] = target_pred.cpu()
        agreement = torch.mean(preds == target_preds, dtype = torch.float).item()*100
        classes = len(np.unique(data.Y))
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
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
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
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
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
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.squeeze().cpu()
        return embeddings

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
    
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x):
        outputs = torch.squeeze(self.linear(x))
        return F.softmax(outputs), x
    
    def get_embedding_dim(self):
        return 784

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__() 
        self.convolutaional_neural_network_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2), 
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2) 
        )
        self.linear_layers = nn.Sequential(
                nn.Linear(in_features=24*7*7, out_features=64),          
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=64, out_features=10)
        )
    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        e1 = x.view(x.size(0), -1)
        x = self.linear_layers(e1)
        return x, e1
    
    def get_embedding_dim(self):
        return 24*7*7
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features,10)
    
    def forward(self,x): 
        e = self.base(x)
        x = self.drop(e.view(-1,self.final.in_features))
        x = self.final(x)
        x = F.softmax(x, dim = 1)
        return x, e
    def get_embedding_dim(self):
        return self.final.in_features
    
class ResNet_small(nn.Module):
    def __init__(self):
        super(ResNet_small, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ), nn.Sequential( 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            nn.Flatten(), 
        )
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        e = self.classifier(x)
        x = self.fc(e)
        
        return x, e
    def get_embedding_dim(self):
        return 512
    

    
    

# For ResNet
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 16, 2)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        e1 = out.view(out.size(0), -1)
        out = self.fc(e1)
        return out, e1

    def get_embedding_dim(self):
        return 64
    

class CNN_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        e = F.relu(self.fc2(x))
        x = self.fc3(e)
        return x, e
    
    def get_embedding_dim(self):
        return 84

class CNN_img(nn.Module):
    def __init__(self):
        
        super(CNN_img, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512)
        )
        self.fc_layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10))


    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        e = self.fc_layer1(x)
        x = self.fc_layer2(e)

        return x, e

    def get_embedding_dim(self):
        return 512
    
