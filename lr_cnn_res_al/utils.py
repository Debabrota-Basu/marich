from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from handlers import MNIST_Handler, ImageNet_Handler, IMGNET_cnn_Handler, SVHN_Handler, CIFAR10_Handler, EMNIST_log_Handler, CIFAR10_log_Handler, STL10_Handler, EMNIST_cnn_Handler, CIFAR10_cnn_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_Imagenet_res_res18, get_Imagenet_res_cnn, get_CIFAR10, get_EMNIST_log, get_CIFAR10_log, get_STL10, get_CIFAR10_cnn, get_EMNIST_cnn, get_IMGNET_cnn
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, ResNet18, LogisticRegression, ResNet, CNN, CNN_img
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool



params = {'EMNIST_log':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 256, 'num_workers': 1},
               'optimizer_args':{'lr': 0.02, 'weight_decay': 0.001},
               'net': 'LogReg',
               'name':'EMNIST_log'},
            'CIFAR10_log':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 256, 'num_workers': 1},
               'optimizer_args':{'lr': 0.02, 'momentum': 0.5},
               'net': 'LogReg',
               'name':'CIFAR10_log'},
            "EMNIST_cnn": {'n_epoch': 10,
                    'train_args':{'batch_size': 64, 'num_workers': 1},
                    'test_args':{'batch_size': 256, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                    'net': 'CNN',
                    'name':'EMNIST_cnn'},
            "CIFAR10_cnn": {'n_epoch': 10,
                    'train_args':{'batch_size': 64, 'num_workers': 1},
                    'test_args':{'batch_size': 64, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                    'net': 'CNN',
                    'name':'CIFAR10_cnn'},
            "Imagenet_res_cnn": {'n_epoch': 5,
                    'train_args':{'batch_size': 64, 'num_workers': 1},
                    'test_args':{'batch_size': 64, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.02, 'momentum': 0.002, 'weight_decay': 0.001},
                    'net': 'CNN_img',
                    'name':'Imagenet_res_cnn'},
            "Imagenet_res_res18": {'n_epoch': 5,
                    'train_args':{'batch_size': 128, 'num_workers': 1},
                    'test_args':{'batch_size': 128, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.02, 'momentum': 0.002, 'weight_decay': 0.001},
                    'net': 'ResNet_small',
                    'name':'Imagenet_res_res18'}
                    }


def get_handler(name):
    if name == "EMNIST_log":
        return EMNIST_log_Handler
    elif name == "CIFAR10_log":
        return CIFAR10_log_Handler
    elif name == "STL10":
        return STL10_Handler
    elif name == "EMNIST_cnn":
        return EMNIST_cnn_Handler
    elif name == "CIFAR10_cnn":
        return CIFAR10_cnn_Handler
    elif name == "Imagenet_res_res18":
        return ImageNet_Handler
    elif name == "Imagenet_res_cnn":
        return ImageNet_Handler
    
def get_dataset(name):
    for k in params:
        params[k]["data"] = name
    if name == "EMNIST_log":
        return get_EMNIST_log(get_handler(name))
    elif name == "CIFAR10_log":
        return get_CIFAR10_log(get_handler(name))
    elif name == "CIFAR10_cnn":
        return get_CIFAR10_cnn(get_handler(name))
    elif name == "EMNIST_cnn":
        return get_EMNIST_cnn(get_handler(name))
    elif name == "Imagenet_res_cnn":
        return get_Imagenet_res_cnn(get_handler(name))
    elif name == "Imagenet_res_res18":
        return get_Imagenet_res_res18(get_handler(name))
    else:
        raise NotImplementedError
    
def get_net(name, device, y_num = None):
    if name == "EMNIST_log" or name == "CIFAR10_log":
        return Net(LogisticRegression, params[name], device, y_num)
    elif name == "EMNIST_cnn" or name == "CIFAR10_cnn":
        return Net(CNN, params[name], device, y_num)
    elif name == "Imagenet_res_cnn":
        return Net(CNN_img, params[name], device, y_num)
    elif name == "Imagenet_res_res18":
        return Net(ResNet18, params[name], device, y_num)
    else:
        raise NotImplementedError

    
def get_params(name):
    return params[name]

def get_strategy(name):
    for k in params:
            params[k]["algo"] = name
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError