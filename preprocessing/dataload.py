import numpy as np
import pickle
import os
from word_embedding import embed_sentences
from nltk import tokenize


def splitAndSanitizeIntoSentences(text):
    sentences = []
    v = text[0]
    subtext = v.text
    sentences = subtext.split(".")
    return sentences, len(sentences)


def parsePerdocs(path):
    f = open(path, "r")
    
    # load all lines into a single string
    fullText = f.read().replace("\n", " ")
    f.close()
    
    # nltk attempt
    #return tokenize.sent_tokenize(fullText) 
    summaries = {} # { docID : summary }
    sumIndex = fullText.find("DOCREF=")
    
    while sumIndex != -1:
        docID = fullText[sumIndex + 8:fullText.find("\"", sumIndex + 9)]
        
        startSum = fullText.find(">", sumIndex)
        endSum = fullText.find("</SUM>", sumIndex)

        text = fullText[startSum + 1:endSum]

        summaries[docID] = text

        sumIndex = fullText.find("DOCREF=", endSum) 
    
    for k in summaries.keys():
        summaries[k] = tokenize.sent_tokenize(summaries[k])

    return summaries
        
def extractText(path):
    f = open(path, "r")

    fullText = f.read().replace("\n", " ")
    f.close()        
    sentences = ""
    textIndex = fullText.find("<TEXT>")
    while textIndex != -1: 
        #sentences.extend(fullText[textIndex + 6 : fullText.find("</TEXT>", textIndex) ])

        #textIndex = fullText.find("<TEXT>", textIndex + 1)
        sentences += fullText[textIndex + 6 : fullText.find("</TEXT>", textIndex) ]
        textIndex = fullText.find("<TEXT>", textIndex + 1)
    
    return tokenize.sent_tokenize(sentences)

def loadTestData(dataRoot):
    '''
    load all of the raw files
    '''
    sentences = {}
    summaries = {}   

    test_data = []
 
    raw_docs = dataRoot + "/docs/"
    walker = os.walk(raw_docs)
    for x in walker:
        path = x[0]
        dirs = x[1]
        files = x[2]    
    
        if len(dirs) != 0:
            continue
    
        for f in files:
            print("file:", path + "/" + f, end="\r") 
            sentences[f] = extractText(path + "/" + f)
        
    raw_summaries = dataRoot + "/summaries/"
    walker = os.walk(raw_summaries)
    for x in walker:
        path = x[0]
        dirs = x[1]
        files = x[2]
        
        if len(dirs) != 0:
            continue 

        for f in files:
            print("summary file:", path + "/" + f, end="\r") 
            tmpSummaries = parsePerdocs(path + "/" + f)
            for k in tmpSummaries.keys():
                summaries[k] = tmpSummaries[k]

   
    size = 0
    hit = 0
    hitsize = 0 
    for s in sentences.keys():
        if s in summaries:
            hit += 1
            hitsize += len(sentences[s])
        size += len(sentences[s]) 
        
    print(size)
    print(hit, "/", len(sentences))
    print(hitsize)

    
    word_embeddings = {}
    count = 0
    max_size = 0
    
    documents_over_190 = 0
    sentences_over_190 = 0
    sentences_removed = 0
    over_190 = False
    

    for s in sentences.keys():
        arr = np.ones((len(sentences[s]), 3), dtype=object) 
        arr[:,0] = "dummy"
        arr[:,1] = np.array(sentences[s])
        #print(arr.shape)
        #print(arr)
        #embedding = embed_sentences(arr, word2vec_limit=None, NUM_WORDS=None )
        embedding = embed_sentences(arr)
        embedding = embedding[0::2]
        for e in embedding:
            if len(e) > max_size:
                max_size = len(e)
            if len(e) > 190:
                sentences_over_190 += 1
                over_190 = True
        if over_190:
            documents_over_190 += 1
            over_190 = False
            count -= len(sentences[s])
            sentences_removed += len(sentences[s])
            continue
        
        count += len(sentences[s])
        #print(embedding.shape)
        #print(len(embedding[0::2]))
        test_data.append((np.array(sentences[s]), np.array(embedding), np.array(summaries[s])))
        print("Finished", count, "of", size,"sentences --", count/size,"%", end='\r')
            
    #print("size of test_data:", len(test_data))
    #print("maximum embedding was", max_size) 
    #print("docs over 190:", documents_over_190)
    #print("sentences over 190:", sentences_over_190)
    #print("removed sentences:", sentences_removed)

    totalCount = 0
    for x in test_data:
        totalCount += len(x[0])
    #print(" full count:", totalCount)

    return test_data
        
        


def loadDUC(dataRoot, summarySize, saliency):
    '''
    parses and returns all of the datasets in the directory

    Params:
        dataRoot        - the root directory for the DUC dataset
                            - ex: extractive-document-summarization/data/DUC2001_Summarization_Documents/data/training
        summarySize     - the size of the summary we wish to explore
                            - ex: 100
        saliency        - callback function to the saliency score
       
    Returns:
        np.array        - [ doc id, sentences, saliency score ]
    '''
    
    # dict to hold the docs and summaries (k: id, v: sentences, saliency scores)
    rawData = {}
    rawSummaries = {}
    
    # running count variable -- keeps track of the total size
    totalSentences = 0 

    # go through all of the roots of the docs
    walker = os.walk(dataRoot)
    for x in walker:
        path = x[0]
        dirs = x[1]
        files = x[2]
        # if docs:
        if len(dirs) == 0: 
            if "perdocs" not in files:
                # for each document
                for f in files:
                    # open, store in the dict as { id : (sentences, []) }
                    try:
                        text = extractText(path + "/" + f)
                        totalSentences += len(text)
                        rawData[f] = text
                    except Exception as e:          
                        print("  ***", path + "/" + f) 
                        print(e)
            else:
                summaries = parsePerdocs(path + "/perdocs")
                for k in summaries.keys():
                    rawSummaries[k] = summaries[k]

    incorrect = 0
    totalSentences = 0
    for k in rawData.keys():
        if k not in rawSummaries:
            print(" key not found in summaries", k)
            incorrect += len(rawData[k])
            continue
        totalSentences += len(rawData[k])
     
    cind = 0
    seen = 0
    nx3output = np.zeros((totalSentences, 3), dtype=object)
    for k in rawData.keys():
        if k not in rawSummaries.keys():
            continue
        seen += 1
        sentences = rawData[k]
        for s in sentences:
            nx3output[cind, 0] = k
            nx3output[cind, 1] = s
            #print(np.array([s]), np.array(rawSummaries[k])[2])
            #print("s", np.array([s]), "\nsummary", np.array(rawSummaries[k]))
            #print("SALIENCY:", saliency(np.array([s]), np.array(rawSummaries[k])))
            try:
#            print()
                nx3output[cind, 2] = saliency(np.array([s]), np.array(rawSummaries[k])) 
                print(" ---- totalSentences:", totalSentences, "cind:", cind)
            except Exception as e:
                print("ERROR: Skipping sentence:", s)
                nx3output[cind, 2] = -1
            cind += 1

    return nx3output

# returns numpy array with the information
def loadFromPickle(fileName):
    f = open(fileName, "rb")
    data = pickle.load(f)
    f.close()
    return data 

def dummy(sentence, summary):
    if sentence in summary:
        return 1
    return 0

def main():
    data = loadTestData("../data/DUC2002_Summarization_Documents")
    #data = loadDUC("../data/DUC2001_Summarization_Documents/data/training", 100, dummy)
    print(data)
    #data = loadFromPickle("sentencesToSaliency.pickle")
    #print(data)

if __name__ == "__main__":
    main()
