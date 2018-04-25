import numpy as np
from keras.models import load_model
from preprocessing.dataload import loadTestData
from preprocessing.rouge import Rouge

def dummy_rouge(sentence_arr, summary_arr, alpha=0.5):
    return np.random.rand()

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
    evals = []
    
    for doc in testing_data: 
        sentences = doc[0]
        
        #x_test_old = doc[1]
        #s1 = x_test_old.shape[0]
        #(s3,s4) = x_test_old[0].shape
        #x_test = np.random.rand(s1,1,378,s4)
        #for i in range(s1) :
        #    x_test[i] = np.array( [ np.pad(x_test_old[i], ((378-s3,0),(0,0)), 'constant') ] )
            

        true_summary = doc[2]
        
        sentences_num = np.random.randint(len(sentences))
        predicted_summary = np.random.choice(sentences.tolist(), sentences_num, replace=False)        
        # print(predicted_summary)
        
        if metric == "ROUGE1" :
            N = 1
        elif metric == "ROUGE2":
            N = 0 
            
        evals.append(dummy_rouge( predicted_summary, true_summary, alpha = N))
        
    return np.mean(evals)
    

def main():
    model = load_model('model.h5')
    #testing_data = dummy_loadTestData()
    testing_data = loadTestData("./data/DUC2002_Summarization_Documents")
    
    rouge1_score = test(model, testing_data, upper_bound=100, metric = "ROUGE1")
    #rouge2_score = test(model, testing_data, upper_bound=5, metric = "ROUGE2")
    print("")
    print(rouge1_score)

if __name__ == "__main__":
    main()

