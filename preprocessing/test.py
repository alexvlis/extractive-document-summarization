import numpy as np
from keras.models import load_model
from dataload import loadTestData
from rouge import Rouge

def dummy_rouge(sentence, summary, alpha=0.5):
    if sentence in summary:
        return 1
    return 0

def dummy_loadTestData():
    testing_data = [ [ np.array(["This sentence is important for doc0." ,
                                 "Such a sentence is irrelevent for doc 0."]), 
                       np.random.rand(2,5,300), 
                       np.array(["This sentence is important for doc0."]) ],
                     [ np.array(["Lol that sentence is awesome for do1." , 
                                 "No way, this is irrelevent"]), 
                       np.random.rand(2,5,300), 
                                np.array(["Lol that sentence is awesome for do1."]) ] ]
    return testing_data

def test(model, testing_data, batch_size = 128, upper_bound = 100, threshold = 0.5, metric = "ROUGE1"):
    """
        Build the actual summaries for test data and evaluate them
        To do: 
            - load the actual x_test (embed test sentences) and y_test (compute rouge score)
        
        Parameters: 
            testing_data           - np.array 
                                        ex: [ doc1, doc2, ... , docn]
                                         where doci = [sentences, x_test, summary]
                                             where sentences = np.array of string
                                                   x_test = np.array of matrices (embedded sentences)
                                                   summaries = np.array of sentences
        
        Returns: 
            eval                   - float between 0 and 1. 
    """   
    rouge = Rouge()
    evals = []
    summaries = []    

    for doc in testing_data: 
        sentences = doc[0]
        
        x_test_old = doc[1]
        s1 = x_test_old.shape[0]
        (s3,s4) = x_test_old[0].shape
        print(s1,s3,s4)
        x_test = np.random.rand(s1,1,190,s4)
        for i in range(s1) :
            x_test[i] = np.array( [ np.pad(x_test_old[i], ((190-s3,0),(0,0)), 'constant') ] )
            

        true_summary = doc[2]
        
        predicted_scores = model.predict(x_test, batch_size=batch_size)
        #argsorted_scores= np.argsort(predicted_scores)
        argsorted_scores = np.argpartition(np.transpose(predicted_scores)[0], 1)
        
        predicted_summary = []
        summary_length = 0
        
        i = 0
        
        while i < len(sentences) and summary_length < upper_bound: 
            sentence = sentences[argsorted_scores[i]]
            #if ( dummy_rouge( sentence , predicted_summary ) < threshold ):
            sentence = np.array([sentence])
            print(sentence, predicted_summary)
            if (rouge.saliency(sentence, np.array(predicted_summary)) < threshold):
                predicted_summary.append(sentence)
                summary_length += len(sentence[0].split())
                
            i+=1
        
        #print(predicted_summary)
        
        if metric == "ROUGE1" :
            N = 1
        elif metric == "ROUGE2":
            N = 0 
            
        #evals.append(dummy_rouge( predicted_summary, true_summary, alpha = N))
        evals.append(rouge.saliency(predicted_summary, true_summary, alpha=N))
        summaries.append((predicted_summary, true_summary))

    print("&& PRINTING SUMMARIES && ")

    for s in summaries:
        print(" -- predicted")
        print(s[0])
        print()
        print(" ++ true")
        print(s[1])
        print(" -------")

    return np.mean(evals)
    

def main():
    model = load_model('../model.h5')
    #testing_data = dummy_loadTestData()
    testing_data = loadTestData("../data/DUC2002_Summarization_Documents")
    print(testing_data)
    
    rouge1_score = test(model, testing_data, upper_bound=100, metric = "ROUGE1")
    #rouge2_score = test(model, testing_data, upper_bound=5, metric = "ROUGE2")
    print(rouge1_score)

if __name__ == "__main__":
    main()

