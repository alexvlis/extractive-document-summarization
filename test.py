import numpy as np
import pickle
from keras.models import load_model
from preprocessing.dataload import loadDUC
from preprocessing.rouge import Rouge

def dummy_prediction():
    return np.array([0.9,0.1, 0.2, 0.8])

def test(model, x_test, y_test, data_test, upper_bound = 100):
    """
        Build the actual summaries for test data.
        To do: 
            - load the actual x_test (embed test sentences) and y_test (compute rouge score)
            - Use score and acc to see our results
            - Use the thershold t when building the summary
            - What to use as a final metric? Do we compare with ROUGE each sentence in the predicted summary to the actual one?  
        
        Returns: 
            predicted_summaries    - np.array of str:  [predicted_summary1, predicted_summary2, ... ]
                        
    """
    # To Do
    # score, acc = model.evaluate(x_test, y_test, batch_size=128)
    
    # To Do
    # predictions = model.predict(x_test, batch_size=128)
    predictions = dummy_prediction()
    
    # Total number of documents
    d = int(data_test[-1,0])
    
    i = 0
    predicted_summaries = np.array([])
    
    # I assumed that doc_id started from 0
    for doc_id in range(d+1):
        # sentences_doc is a list containng all the sentences of a document with their prediected saliency score [score,sentence]
        sentences_doc = []
        while i < len(data_test) and int(data_test[i,0]) == doc_id:
            sentences_doc.append([ float(predictions[i]), data_test[i,1] ])
            i += 1
        
        sentences_doc = np.array(sentences_doc)
        #sort the sentences of a doc by saliency score
        sentences_doc = sentences_doc[sentences_doc[:,0].argsort()]
        #Reverse the order (decreasing order)
        sentences_doc = sentences_doc[::-1]
        
        predicted_summary = ""
        
        
        ind = 0
        # While the summary is not full
        # I chose to go slitly beyond the upper_bound (one sentence beyond) but we can decide to stay under it
        while len(predicted_summary.split()) < upper_bound:
            sentence = sentences_doc[ind,1]
            predicted_summary = predicted_summary+sentence
        
        predicted_summaries = np.append(predicted_summaries,predicted_summary)
    
    return predicted_summaries

def main():
    model = load_model('model.h5')
    
    #To Do: load the embedded sentences 
    x_test = []
    #To Do: Compute the saliency scores for testing
    y_test = []
    
    #rougeSaliency = Rouge()
    #data_test = loadDuc("../data/DUC2002_Summarization_Documents/data/testing", upper_bound, rougeSaliency.saliency)
    
    data_test = np.array([ [0, "This sentence is important for doc0."], [0,"Such a sentence is irrelevent for doc 0."], [1, "Lol this sentence makes no sense for doc1."], [1,"However this one is crucial for doc1."] ])
    predicted_summaries = test(model, x_test, y_test, data_test, upper_bound = 5) 
    print(predicted_summaries)

if __name__ == "__main__":
    main()

