# Marich
We present three notebooks for reproduction of the experiments. The three notebooks contain model extraction using MARICH and other algorithms using three datasets, on various kinds of models.
The data used for training the bert target model is bbc-text.csv.
ag_unlabeled.csv contains ag_news data but the labels are changed, i.e. the dataset contains $x,f^T(x)$. Getting this $(x,f^T(x))$ pair has been time taking and thus we provide the file assuming that a new $f^T$ will not vary a lot when trained by user.

PS: User may need to change paths top save and load models from in the notebooks according to their structures.
