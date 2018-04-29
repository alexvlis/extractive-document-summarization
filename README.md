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

## Example Summaries: ##
| Ground Truth Summary           | Model Generated Summary       | 
| ----------------|:-------------:| 
| "Rice was named the game’s most valuable player. Joe Mon- tana’s 10-yard touchdown pass to John Taylor with 34 seconds left provided the winning margin. The victory was achieved through the brilliance of Montana and Jerry Rice, the wide receiver who caught 12 passes for 222 yards, both Super Bowl records. In a dramatic finish, the San Francisco 49ers football team won the Su- per Bowl on Sunday with a 20- 16 victory over the Cincinnati Bengals. The winning score came at the end of a 92-yard drive, engineered by Montana, whose touchdown pass gave him a Su- per Bowl record with 357 yards passing." | "The San Francisco 49ers won the Super Bowl Sunday with a dramatic 20-16 victory over the Cincinnati Bengals. Joe Mon- tana’s 10-yard touchdown pass to John Taylor with 34 seconds remaining provided the win. The pass gave Montana a Super Bowl record with 357 yards passing. The victory was achieved through the brilliance of Montana and Jerry Rice, the wide receiver who caught 12 passes for 222 yards, both Super Bowl records. Rice was named the game’s most valuable player. It was the fifth straight win for a National Foot- ball Conference team and the third Super Bowl win for Coach Bill." | 
