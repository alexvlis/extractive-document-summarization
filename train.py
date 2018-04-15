# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:01:27 2018

@author: leo
"""

import pandas as pd
import numpy as np
import gensim

from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers import SpatialDropout1D, Flatten, MaxPooling1D , Input
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import Conv1D
from keras import backend as K

# ---------------------- Parameters start -----------------------
model_type = "CNN-rand"  # CNN-rand|CNN-word2vec
summarization = "single" # single|multi

# Model Hyperparameters
word_limit = 50000
embedding_dim = 300
window_size = 3
num_filters = 400
dropout_prob = 0.5
max_features = 10000

# Training parameters
batch_size = 64
nb_epoch = 20

# ---------------------- Parameters end -----------------------




# ---------------------- Preprocessing start -----------------------





# Prepare embedding layer weights and convert inputs for static model
if model_type = "CNN-word2vec":
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=word_limit)
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
     x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
     x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
     print("x_train static shape:", x_train.shape)
     print("x_test static shape:", x_test.shape)
    
elif model_type == "CNN-rand":
    embedding_weights = None

else:
    raise ValueError("Unknown model type")
    
# ---------------------- Preprocessing end -----------------------
    

# ---------------------- Model start -----------------------

# Build model
print('Build model...')
model = Sequential()
model.add(Embedding(max_features,300))
# we add a Convolution 1D, which will learn nb_filter
# word group filters of size window_size:
model.add(Conv1D(filters=num_filters,
                         kernel_size=window_sizedow_size,
                         padding="valid",
                         activation="relu",
                         strides=1))
#Max pooling
def max_1d(X):
    return K.max(X, axis=1)

model.add(Lambda(max_1d, output_shape=(nb_filter,)))

#Fully Connected + Dropout + softmax
model.add(Dropout(dropout_prob)) 
model.add(Dense(1, activation='sigmoid'))

# ---------------------- Model end -----------------------

# ---------------------- Training/Testing start-----------------------

#In addition, an l2−norm constraint of the weights w_r is imposed during training as well

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
# ---------------------- Training/Testing end-----------------------