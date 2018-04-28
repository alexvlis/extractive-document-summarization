import numpy as np
from keras.models import load_model
from dataload import loadTestData
from rouge import Rouge
import nltk

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

def test(model, testing_data, batch_size = 128, upper_bound = 100, threshold = 1, metric = "ROUGE1"):
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
    r1evals = []
    r2evals = []
    summaries = []    

    r1bestPred = []
    r1bestTrue = []
    r1worstPred = []
    r1worstTrue = []
    r1bestScore = -1.0
    r1worstScore = 1.1
    
    r2bestPred = []
    r2bestTrue = []
    r2worstPred = []
    r2worstTrue = []
    r2bestScore = -1.0
    r2worstScore = 1.1
    
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
            #print(sentence, predicted_summary)
            predicted_summary.append(sentence)
            summary_length += len(nltk.word_tokenize(sentence[0]))
                
            i+=1
        
        #print(predicted_summary)
        
        #if metric == "ROUGE1" :
        #    N = 1
        #elif metric == "ROUGE2":
        #    N = 0 
            
        #evals.append(dummy_rouge( predicted_summary, true_summary, alpha = N))
        r1score = rouge.saliency(predicted_summary, true_summary, alpha=1)
        r2score = rouge.saliency(predicted_summary, true_summary, alpha=0)

        if r1score > r1bestScore:
            r1bestScore = r1score
            r1bestPred = predicted_summary
            r1bestTrue = true_summary
        if r1score < r1worstScore:
            r1worstScore = r1score
            r1worstPred = predicted_summary
            r1worstTrue = true_summary
        
        if r2score > r2bestScore:
            r2bestScore = r2score
            r2bestPred = predicted_summary
            r2bestTrue = true_summary
        if r2score < r2worstScore:
            r2worstScore = r2score
            r2worstPred = predicted_summary
            r2worstTrue = true_summary

        r1evals.append(r1score)
        r2evals.append(r2score)

        #evals.append(rouge.saliency(predicted_summary, true_summary, alpha=N))
        summaries.append((predicted_summary, true_summary))

    print("&& PRINTING BEST AND WORST SUMMARIES && ")
   
    print("ROUGE 1 --") 
    print("BEST:")
    print(" score:", r1bestScore)
    print(" predicted:", r1bestPred)
    print(" true:", r1bestTrue)
    
    print()
    print("WORST:")
    print(" score:", r1worstScore)
    print(" predicted:", r1worstPred)
    print(" true:", r1worstTrue)
    print()

    print("ROUGE 2 --") 
    print("BEST:")
    print(" score:", r2bestScore)
    print(" predicted:", r2bestPred)
    print(" true:", r2bestTrue)
    
    print()
    print("WORST:")
    print(" score:", r2worstScore)
    print(" predicted:", r2worstPred)
    print(" true:", r2worstTrue)

    print(" *--*--*--*--*--*--*--*")

    return np.mean(r1evals), np.mean(r2evals)
    

def main():
    model = load_model('../model-nfilt-200.h5')
    #testing_data = dummy_loadTestData()
    #testing_data = loadTestData("../data/DUC2002_Summarization_Documents")
    testing_data = loadTestData("../data/test_subset")
    print(testing_data)
    
    rouge1_score, rouge2_score = test(model, testing_data, upper_bound=100, metric = "ROUGE1")
    #rouge2_score = test(model, testing_data, upper_bound=100, metric = "ROUGE2")
    print("ROUGE1:",rouge1_score)
    print("ROUGE2:",rouge2_score)

if __name__ == "__main__":
    main()

