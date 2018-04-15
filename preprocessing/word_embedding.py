# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:21 2018

@author: leock
"""

import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def embed_sentences(sentences, word2vec_limit = 50000 , NUM_WORDS=20000):   
    '''
    Embed sentences

    Params:
        sentences        - a list of sentences
                            - ex: ["It's the first sentence and I feel good about!","it is the second sentence"]
        word2vec_limit   - int: number of words used in the word embedding provided by Google
                            - ex: 50000
        NUM_WORDS        - int: The maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
                            - ex: 20000
       
    Returns:
        np.array        - 3D array: 1D for the sentences, 1D for the words and 1D for the word2vec dimensions. 
    '''
    
    #Load Google pre-trained words as a model
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True, limit=word2vec_limit)
    #Convert the model as a dictionnary word_vectors["hello"] will return a vector like [0.3, 3, ... , -4]
    word_vectors = embedding_model.wv
    print("Embedding for 'hello': ", word_vectors["hello"], "\n")
    
    # Tokenize the sentences, that is to say convert the 2 sentences ["It's the first sentence!","It is the second sentence"] to 2 sequences [[1 4 2 5 3],[1 6 2 7 3]]
    # It handles the ennoying cases (punctuation, Upper cases, etc...)
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)
    print("Padded_sequences: ", padded_sequences, "\n")   
    
    #Produce a dictionnary mapping words to tokens e.g. {'it': 1, 'the': 2, 'sentence': 3, 's': 4, 'first': 5, 'is': 6, 'second': 7}
    word_index = tokenizer.word_index
    print("word_index: " , word_index , "\n" )
    
    #Build a dictionnary mapping tokens to vectors e.g. {'1': [2, ... , -3] ; '2': [4, ... , 0.8] ; ... }
    embedding_weights = {key: embedding_model[word] if word in word_vectors.vocab else
                              np.random.uniform(-0.25, 0.25, word_vectors.vector_size)
                        for word, key in word_index.items()}
    # Add the token "0", used for padding
    embedding_weights[0] = np.zeros(word_vectors.vector_size)
   
    print("Embedding weights: " , embedding_weights , "\n")
    
    #Build a 3D array: 1D for the sentences, 1D for the words and 1D for the word2vec dimensions. 
    embedded_sentences = np.stack([np.stack([embedding_weights[token] for token in sentence]) for sentence in padded_sequences])
    
    del embedding_model
    
    return embedded_sentences


if __name__ == "__main__":
    # For debugging purpose
    embedded_sentences = embed_sentences(["It's the first sentence and I feel good about!","it is the second sentence"])
    print(embedded_sentences)
