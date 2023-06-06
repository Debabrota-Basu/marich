# Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack using Public Data

Marich aims to extract models using public data with two different motives:
1. Distributional Equivalence and
2. Query Efficiency

To achieve these two Marich uses an active learning algorithm to query and extract the target models $(f_T)$. We assume that only the labels (not the probabilities) are available from the target models. The extracted models are trained on the selected $x$'s and $\hat{y}$'s obtained from the target models.

There are 4 folders:
bert_al: Contains K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling codes for BERT experiments
lr_cnn_res_al: Contains K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling codes for experiments on Logistic Regression, CNN and ResNet
bert_marich: Contains Marich codes for BERT experiments
lr_cnn_res_marich: Contains Marich codes for experiments on Logistic Regression, CNN and ResNet

The jupyter notebooks provided in the folders act as demo for the users.

To experiment with new data, one needs to:
1. In data.py file, add compatible get_DATA function. Follow the structure of the existing get_DATA functions.
2. In handlers.py file add a compatible Handler class. Follow the structure of the existing Handler classes.
3. In case of Marich new data input is to be given following the jupyter notebooks.

To experiment with new models, one needs to:
1. Add the corresponding model to the nets.py file. For the active learning algorithms, other than Marich, one must remember to modify the model to have a forward method returning the output and a preferred embedding, and have a method to return the embedding dimension.


For the K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling experiments, we have modified and used the codes from https://arxiv.org/pdf/2111.15258.pdf


@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
