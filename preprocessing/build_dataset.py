from rouge import Rouge
from dataload import loadDUC
from word_embedding import embed_sentences 

import numpy as np
import pandas as pd
import pickle


def buildData(datasetRoot, saliency):
    data = loadDUC(datasetRoot, 100, saliency)  
    f = open("sentencesToSaliency.pickle", "wb")
    pickle.dump(data, f)
    #print("saved sentences to", fileName)
    return embed_sentences(data) 

def saveData(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename)

def main():
    rougeSaliency = Rouge() 
    print("got rouge")
    #data = buildData("../data/DUC2001_Summarization_Documents/data/training", rougeSaliency.saliency)
    data = buildData("../data/subset/data/training", rougeSaliency.saliency)
    print("built data")

    fileName = "wordEmbeddingsToSaliency.pickle"
    f = open(fileName, "wb")
    print("saving data to", fileName)
    pickle.dump(data, f)
    print("Saved data!")

    #saveData("duc2001-dataset.csv", data)
    #print("saved")
    
    

if __name__ == "__main__":
    main()

