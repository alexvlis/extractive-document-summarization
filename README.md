# Extractive Document Summarization
Extractive Document Summarization Based on Convolutional Neural Networks.

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
