from rouge import Rouge
from dataload import loadDUC, loadFromPickle
from word_embedding import embed_sentences 

import numpy as np
import pandas as pd
import pickle
import sys


def buildData(datasetRoot, saliency):
    data = loadDUC(datasetRoot, 100, saliency)  
    f = open("sentencesToSaliency.pickle", "wb")
    pickle.dump(data, f)
    f.close()
    #print("saved sentences to", fileName)
    return embed_sentences(data, NUM_WORDS=None, word2vec_limit=None) 

def saveData(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename)

def main():
    rougeSaliency = Rouge() 
    print("got rouge")
   
    # use if you need to start from scratch 
    data = buildData("../data/DUC2001_Summarization_Documents/data/training", rougeSaliency.saliency)    
    #data = buildData("../data/subset/data/training", rougeSaliency.saliency)

    print("built data")

    # use if you have the pickle file
    data = loadFromPickle("sentencesToSaliency.pickle")
    data = embed_sentences(data)
    print(data)
    print("loaded data:", data.shape)
    print("size:", sys.getsizeof(data))

    #fileName = "wordEmbeddingsToSaliency1.pickle"
    
    #f = open(fileName, "wb")
    #print("saving data to", fileName)
    
    #pickle.dump(data[:int(len(data)/2)], f)
    #f.close()
    #print("saved first round")

    #fileName = "wordEmbeddingsToSaliency2.pickle"
    #f = open(fileName, "wb")
    #print("saving second set of data to", fileName)
    #pickle.dump(data[int(len(data)/2):], f)
    #f.close()

    #print("Saved data!")

    num_parts = 8
    fileName = "wordEmbeddingsToSaliency"
    start = 0
    
    for i in range(num_parts):
        print("writing part,",i + 1)
        f = open(fileName + str(i + 1) + ".pickle", "wb")
        if i < num_parts - 1:
            pickle.dump(data[start:(i + 1) * len(data)//num_parts], f)
        else:
            pickle.dump(data[start:], f)
        f.close()
        start = (i + 1) * len(data)//num_parts
        print(start)

    
    

if __name__ == "__main__":
    main()

