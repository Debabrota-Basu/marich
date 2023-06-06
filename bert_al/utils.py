from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from handlers import BERT_Handler
from data import get_BERT
from nets import BertClassifier, Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

# params = 
params = {'BERT':
              {'n_epoch': 3, 
               'train_args':{'batch_size': 4, 'num_workers': 1},
               'test_args':{'batch_size': 8, 'num_workers': 1},
               'optimizer_args':{'lr': 5e-6, 'weight_decay': 0.001},
               'net': 'BERT',
               'name':'BERT'}}


def get_handler(name):
    if name == "BERT":
        return BERT_Handler
    
def get_dataset(name):
    for k in params:
        params[k]["data"] = name
    if name == "BERT":
        return get_BERT(get_handler(name))
    else:
        raise NotImplementedError
    
def get_net(name, device, y_num = None):
    if name == "BERT":
        return Net(BertClassifier, params[name], device, y_num = 5)
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