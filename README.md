# Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack using Public Data

Marich aims to extract models using public data with three motives:
1. Distributional Equivalence of Extracted Prediction Distribution
2. Max-Information Extraction of the Target Model
3. Query Efficiency

To achieve these goals, Marich uses an active learning algorithm to query and extract the target models $(f_T)$. We assume that only the labels (not the probabilities) are available from the target models. The extracted models $(f_E)$ are trained on the selected $x$'s and $\hat{y}$'s obtained from the target models.

The attack framework is as given below:

![My Image](figures/attack_framework.png)

## Resources

Paper: https://arxiv.org/abs/2302.08466

Talk at PPAI Workshop at AAAI, 2023: [Slides](https://debabrota-basu.github.io/slides/marich_ppai23.pdf)


## Summary of Results

### Accuracy of Extracted Model
The accuracies of competing active learning methods are shown along with Marich to present a comparison:

<p align="center">
<img src="figures/legend_acc.png" width = 400>
</p>
    
 <p align="center">
<img src="figures/LR_emnist.png" width="395" title="LR extracted using EMNIST"/> <img src="figures/LR_cifar.png" width="395" title="LR extracted using CIFAR10"/> <img src="figures/bert_acc.png" width="395" title="BERT extracted using AGNEWS"/> <img src="figures/Res_CNN.png" width="395" title="ResNet extracted using ImageNet"/>
</p>

The accuracy curves shown above are respectively for:
1. Logistic regression model trained on MNIST dataset extracted using another Logistic regression model with EMNIST queries.
2. Logistic regression model trained on MNIST dataset extracted using another Logistic regression model with CIFAR10 queries.
<!--3. CNN trained on MNIST dataset extracted using another CNN with EMNIST queries.-->
3. BERT trained on BBC News dataset extracted using another BERT with AG News queries.
4. ResNet trained on CIFAR10 dataset extracted using a CNN with ImageNet queries.

### Distributional Equivalence of Prediction Distributions
Next we present the kl divergence between the outputs of the extracted models and the target models to compare the distributional equivalence of the models extracted by different algorithms. This is done on a separate subset of the training domain data.

<p align="center">
<img src="figures/kl_lr_emnist.png" width="395" title="LR extracted using EMNIST"/> <img src="figures/kl_log_cifar.png" width="395" title="LR extracted using CIFAR10"/> <img src="figures/kl_bert.png" width="395" title="BERT extracted using AGNEWS"/> <img src="figures/kl_res_cnn.png" width="395" title="ResNet extracted using ImageNet"/>
</p>


### Membership Inference with Extracted Models
The order of the extraction set ups are same as mentioned for the accuracies.
The table below shows a portion of the results obtained during our experiments:

<img src="figures/table.png">

## How to Run Marich?

There are 4 folders:

**bert_al**: Contains K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling codes for BERT experiments

**lr_cnn_res_al**: Contains K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling codes for experiments on Logistic Regression, CNN and ResNet

**bert_marich**: Contains Marich codes for BERT experiments

**lr_cnn_res_marich**: Contains Marich codes for experiments on Logistic Regression, CNN and ResNet

The jupyter notebooks provided in the folders act as demo for the users.

To experiment with new data, one needs to:
1. In data.py file, add compatible get_DATA function. Follow the structure of the existing get_DATA functions.
2. In handlers.py file add a compatible Handler class. Follow the structure of the existing Handler classes.
3. In case of Marich new data input is to be given following the jupyter notebooks.

To experiment with new models, one needs to:
1. Add the corresponding model to the nets.py file. For the active learning algorithms, other than Marich, one must remember to modify the model to have a forward method returning the output and a preferred embedding, and have a method to return the embedding dimension.

For the K-Center, Least Confidence, Margin Sampling, Entropy Sampling and Random Sampling experiments, we have modified and used the codes from Huang, Kuan-Hao. "Deepal: Deep active learning in python." 2021. (Link: https://arxiv.org/pdf/2111.15258.pdf)

## Reference

If you use or study any part of this repository, please cite it as:
```
@article{karmakar2023marich,
  title={Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack using Public Data},
  author={Karmakar, Pratik and Basu, Debabrota},
  journal={arXiv preprint arXiv:2302.08466},
  year={2023}
}
```
