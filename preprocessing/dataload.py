import numpy as np
import pickle
import os
import re
from word_embedding import embed_sentences
from nltk import tokenize
from rouge import Rouge


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
    summaries = {} # { docID : summary }
    sumIndex = fullText.find("DOCREF=")
   
    # gets all of the summaries and stores them in the appropriate places 
    while sumIndex != -1:
        docID = fullText[sumIndex + 8:fullText.find("\"", sumIndex + 9)]
        
        startSum = fullText.find(">", sumIndex)
        endSum = fullText.find("</SUM>", sumIndex)

        text = fullText[startSum + 1:endSum]
        text = text.replace("<P>", " ")
        text = text.replace("</P>", " ")

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
    # extracts the text in the documents
    # looks for the <TEXT> and </TEXT> tags
    while textIndex != -1: 
        sentences += fullText[textIndex + 6 : fullText.find("</TEXT>", textIndex) ]
        textIndex = fullText.find("<TEXT>", textIndex + 1)

    #old = sentences
    sentences = sentences.replace("<P>", " ")
    sentences = sentences.replace("</P>", " ")
    #xmlRegex = re.compile("<.*?>.*?</.*?>|<.*?/>", re.IGNORECASE)
    #sentences = xmlRegex.sub("<.*?>.*?</.*?>|<.*?/>", sentences)
    #print(old, sentences)
    sentences = sentences.replace(";", " ")

    return tokenize.sent_tokenize(sentences)

def _countMatchingTestData(sentences, summaries):
    size = 0
    hit = 0
    hitsize = 0 
    for s in sentences.keys():
        if s in summaries:
            hit += 1
            hitsize += len(sentences[s])
        size += len(sentences[s]) 
    return size, hit, hitsize

def _createEmbeddedTestData(sentences, summaries):

    size, hit, hitsize = _countMatchingTestData(sentences, summaries)

    test_data = []
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
        test_data.append((np.array(sentences[s]), np.array(embedding), np.array(summaries[s])))
        print("Finished", count, "of", size,"sentences --", count/size,"%", end='\r')
    return test_data
    
def loadTestData(dataRoot):
    '''
    load all of the test files -- specifically built for DUC2002
    '''
    sentences = {}
    summaries = {}   

    test_data = []

    # gets the raw documents 
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
        
    # fetches the summaries
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

   
    size, hit, hitsize = _countMatchingTestData(sentences, summaries)   

    embedded = _createEmbeddedTestData(sentences, summaries)
    return embedded
    
           
      

def _calculateNumberOfSentences(summaries, data):
    '''
    Calculates the number of sentences in the data
    useful so that we dont need to reshape the numpy array a large number of times
    
    it also verifies that both of the keys appear in the dictionaries -- only counting sentences
        of documents that have matching summaries
    '''
    incorrect = 0
    totalSentences = 0
    for k in data.keys():
        if k not in summaries:
            print(" key not found in summaries", k)
            incorrect += len(data[k])
            continue
        totalSentences += len(data[k])
    return totalSentences
      
 
def _packageInNumpyArray(summaries, data, saliency):
    '''
    Takes in the summaries and the data and then creates a numpy array [[docID, sentence1, summary],
                                                                        [docID, sentence2, summary],
                                                                                   ...          
    '''
    
    totalSentences = _calculateNumberOfSentences(summaries, data)
    cind = 0
    seen = 0
    skipped = 0
    parsed = 0
    nx3output = np.zeros((totalSentences, 3), dtype=object)
    for k in data.keys():
        if k not in summaries.keys():
            continue
        seen += 1
        sentences = data[k]
        summary = np.array(summaries[k])
        for s in sentences:
            nx3output[cind, 0] = k
            nx3output[cind, 1] = s
            try:
                nx3output[cind, 2] = saliency(np.array([s]), summary) 
                parsed += 1
                print(" ---- totalSentences:", totalSentences, "cind:", cind)
            except Exception as e:
                skipped += 1
                print("ERROR: Skipping sentence:", s)
                nx3output[cind, 2] = -1
            cind += 1
    print("parsed:", parsed, "Skipped:", skipped)
    return nx3output




def loadDUC(dataRoot, summarySize, saliency):
    '''
    parses and returns all of the datasets in the directory -- built specifically for DUC2001

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
        
        # gets all of the info about the specific directory we're in
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
    nx3output = _packageInNumpyArray(rawSummaries, rawData, saliency)
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
    r = Rouge()
    testdata = loadTestData("../data/test_subset")
    #data = loadDUC("../data/subset/data/training", 100, r.saliency)
    print(testdata)
    #data = loadFromPickle("sentencesToSaliency.pickle")
    #print(data)

if __name__ == "__main__":
    main()
