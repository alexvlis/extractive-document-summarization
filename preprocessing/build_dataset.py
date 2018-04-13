from rouge import Rouge
from dataload import loadDUC
from word_embedding import embed_sentences 

import numpy as np
import pandas as pd


def buildData(datasetRoot, saliency):
    data = loadDUC(datasetRoot, 100, saliency)  
    return embed_sentences(data) 

def saveData(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename)

def main():
    rougeSaliency = Rouge() 
    print("got rouge")
    data = buildData("../data/DUC2001_Summarization_Documents/data/training", rougeSaliency.saliency)
    print("built data")
    saveData("duc2001-dataset.csv", data)
    print("saved")
    
    

if __name__ == "__main__":
    main()

