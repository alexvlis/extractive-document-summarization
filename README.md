# Extractive Document Summarization using CNNs
Extractive Summarization is a method, which aims to automatically generate summaries of documents through the extraction of sentences in the text. The specific model we implemented is a regression process for sentence ranking on the [DUC Dataset](https://duc.nist.gov/data.html). The architecture of this method consists of a convolution layer followed by a max-pooling layer, on top of a pre-trained word2vec mapping. We implemented this new proposed method and perform experiments on single-document extractive summarization.

## Installation: ##
```
git clone --recurse-submodules https://github.com/alexvlis/extractive-document-summarization.git
cd extractive-document-summarization/
conda create --name <env> --file requirements.txt
source activate <env> 
pyrouge_set_rouge_path /global/pathto/extractive-document-summarization/preprocessing/pyrouge/tools/ROUGE-1.5.5/

source deactivate
```

## Download Google's Word2vec: ##
```
cd word2vec/
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

## Data Preprocessing: ##

From the ```preprocessing``` directory, execute ```python build_dataset.py```. This will create multiple pickle files. 

The first is a pickle file of the sentences to the saliency scores (```sentencesToSaliency.pickle```).

The rest are pickle files of the word embeddings to the saliency scores for each sentences (```wordEmbeddingsToSaliency.pickle```).

## Training: ##
```
python train.py
```
![alt text](https://github.com/alexvlis/extractive-document-summarization/blob/master/figures/training-conv-190.png "Logo Title Text 1")

## Results: ##
```
cd preprocessing/
python test.py
```
| Model           | ROUGE-1       | ROUGE-2  |
| ----------------|:-------------:| --------:|
| [Zhang et al.](https://ieeexplore.ieee.org/document/7793761/?reload=true)    | 48.62%        |   21.99% |
| Our Implementation| 47.51%        |   22.41% |
| Random Baseline | 32.14%        |   11.39% |
