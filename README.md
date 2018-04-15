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


## Data Preprocessing: ##

From the ```preprocessing``` directory, execute ```python3 build_dataset.py```.
This will dump two pickle files. 

The first is a pickle file of the sentences to the saliency scores (```sentencesToSaliency.pickle```).

The second is a pickle file of the word embeddings to the saliency scores for each sentences (```wordEmbeddingsToSaliency.pickle```).
