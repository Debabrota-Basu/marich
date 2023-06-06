import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialBIM(Strategy):
    def __init__(self, dataset, net, eps=0.05):
        super(AdversarialBIM, self).__init__(dataset, net)
        self.eps = eps

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0).to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        self.net.clf.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")

        out, e1 = self.net.clf(nx+eta)
        py = out.max(0)[1]
        ny = out.max(0)[1]
        while py.item() == ny.item():
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out, e1 = self.net.clf(nx+eta)
            py = out.max(0)[1]

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.cpu()
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.clf.cuda()

        return unlabeled_idxs[dis.argsort()[:n]]


