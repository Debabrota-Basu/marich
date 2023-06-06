import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialDeepFool(Strategy):
    def __init__(self, dataset, net, max_iter=50):
        super(AdversarialDeepFool, self).__init__(dataset, net)
        self.max_iter = max_iter

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0).to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(torch.device("cuda") if torch.cuda.is_available() else "cpu")

        out, e1 = self.net.clf(nx+eta)
        n_class = out.shape[1] if len(out.shape)==2 else out.shape[0] 
        py = out.max(0)[1].item()
        ny = out.max(0)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[i] - out[py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.cpu().numpy().flatten()+1e-6)

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi.cpu()
                    value_l = value_i

            eta += ri.cuda().clone()
            nx.grad.data.zero_()
            out, e1 = self.net.clf(nx+eta)
            py = out.max(0)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.clf.cuda()

        return unlabeled_idxs[dis.argsort()[:n]]


