import numpy as np
import os


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
    #print(fullText)
    summaries = {} # { docID : summary }
    sumIndex = fullText.find("DOCREF=")
    while sumIndex != -1:
        docID = fullText[sumIndex + 8:fullText.find("\"", sumIndex + 9)]
        
        startSum = fullText.find(">", sumIndex)
        endSum = fullText.find("</SUM>", sumIndex)

        text = fullText[startSum + 1:endSum]

        summaries[docID] = text.split(".")

        sumIndex = fullText.find("DOCREF=", endSum) 

    
    return summaries
        
def extractText(path):
    f = open(path, "r")

    fullText = f.read().replace("\n", " ")
    f.close()        

    sentences = []
    textIndex = fullText.find("<TEXT>")
    while textIndex != -1: 
        sentences.extend(fullText[textIndex + 6 : fullText.find("</TEXT>", textIndex) ].split("."))

        textIndex = fullText.find("<TEXT>", textIndex + 1)
    return sentences
    

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
        #print(x)
        # if docs:
        if len(dirs) == 0: 
            if "perdocs" not in files:
                # for each document
                for f in files:
                    # open, store in the dict as { id : (sentences, []) }
                    try:
                        #tree = xet.parse(path + "/" + f).getroot()
                        #sents = []
                        #text = tree.findall("TEXT")
                        text = extractText(path + "/" + f)
                        #rawData[f], s = splitAndSanitizeIntoSentences(text)
                        totalSentences += len(text)
                        rawData[f] = text
                    except Exception as e:          
                        print("  ***", path + "/" + f) 
                        print(e)
            else:
                #totalSentences += 1
                #print("  perdocs:", path)
                # look in perdocs
                print(path + "/perdocs")
                #tree = xet.parse(path + "/perdocs")
                summaries = parsePerdocs(path + "/perdocs")
                for k in summaries.keys():
                    rawSummaries[k] = summaries[k]
                # for each docID:
                #for summary in tree.findall("SUM"):
                #    print(summary.attrib)
                    # compute saliency score
                    # store in dict as { id : (sentences, saliency scores) }

    incorrect = 0
    totalSentences = 0
    for k in rawData.keys():
        if k not in rawSummaries:
            print(" key not found in summaries", k)
            incorrect += len(rawData[k])
            #totalSentences -= len(rawData[k])
            continue
        totalSentences += len(rawData[k])
        #print(" ", k, "(" + str(len(rawData[k])) + ", " + str(len(rawSummaries[k])) + ")")
    #print(" total incorrect:", incorrect, "total sentences:", totalSentences)
     
    cind = 0
    seen = 0
    nx3output = np.zeros((totalSentences, 3), dtype=object)
    #print(summaries.keys())
    for k in rawData.keys():
        if k not in rawSummaries.keys():
            continue
        seen += 1
        sentences = rawData[k]
        for s in sentences:
            nx3output[cind, 0] = k
            nx3output[cind, 1] = s
            nx3output[cind, 2] = saliency(s, rawSummaries[k]) 
            cind += 1
    #print(cind, totalSentences)
    #print(seen, len(rawData))


    return nx3output
        # get the docs
        # get the summaries -- in perdocs (size only 100?) 
        
        # for each sentence
            # compute saliency score

        # increment running number of sentences by |docs|

    # create running count x 3 np.array
    # [ doc ID | sentence | saliency ]

    # for each k
        # get the dict k
        # for each sentence in dict[k]
            # store in output array [ k | sentence | saliency score ] 
    

def dummy(sentence, summary):
    if sentence in summary:
        return 1
    return 0

def main():
    data = loadDUC("../data/DUC2001_Summarization_Documents/data/training", 100, dummy)
    print(data)

if __name__ == "__main__":
    main()
