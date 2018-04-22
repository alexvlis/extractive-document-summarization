"""
Created on Sat Mar 17 11:01:27 2018

@author: leo
"""
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras.optimizers import Adadelta

from preprocessing.word_embedding import rand_embed_sentences

from sklearn.utils import shuffle

# Build model
def build_model(input_shape, conv_window_size, num_filters, reg, dropout, word2vec = True, max_token = None, sequence_len= 190):
    """
    If random init
        max_token is the vocabulary size
        sequence_len is the number of words in the largenst sentence
    """
    
    model = Sequential()
    #model.add(Embedding(max_features,300))

    # we add a Convolution 1D, which will learn num_filters
    # word group filters of size conv_window_size:
    if (not(word2vec)) :
        model.add(Embedding(max_token,300, input_length=sequence_len))
        input_shape = (1, sequence_len, 300)
    
    
    model.add(Conv2D(input_shape=input_shape,
                        filters=num_filters,
                        kernel_size=(1, conv_window_size),
                        padding="valid",
                        activation="relu",
                        strides=1,
                        data_format='channels_first'))
    
    model.add(MaxPooling2D(pool_size=(num_filters, 1)))

    #Fully Connected + Dropout + sigmoid
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg)))
    
    #In addition, an l2−norm constraint of the weights w_r is imposed during training as well

    model.compile(loss='binary_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['mae'])
    return model

def train(model, x_train, y_train, val_train_ratio=0.2, epochs=1000, batch_size=128):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_train_ratio,
                        shuffle=False,
                        verbose=1)
    return history

def load_data():
    print("loading pickle files...")
    data1 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency1.pickle", "rb"))
    data2 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency2.pickle", "rb"))
    data3 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency3.pickle", "rb"))
    data4 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency4.pickle", "rb"))
    data5 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency5.pickle", "rb"))
    data6 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency6.pickle", "rb"))
    data7 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency7.pickle", "rb"))
    data8 = pickle.load(open("preprocessing/wordEmbeddingsToSaliency8.pickle", "rb"))

    print("concatenating data...")
    data = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8), axis=0)

    print("extracting x and y...")
    x = data[::2]
    y = data[1::2]
    del data

    print("converting x to np tensor...")
    x = np.dstack(x)
    x = np.rollaxis(x, -1)
    x = np.expand_dims(x, axis=1)

    mask = y==-1

    print("removing -1s...")
    x = x[~mask, :]
    y = y[~mask]

    print("data loaded.")
    x, y = shuffle(x, y)
    return x, y

def dummy_load_data(data):
    x,y = rand_embed_sentences(data)
    return (x,y)

def main():
    # Model Hyperparameters
    conv_window_size = 300
    num_filters = 400
    reg = 0.01
    dropout = 0.5
    
    # Training parameters
    epochs = 25
    batch_size = 128
    test_train_ratio = 0.2
    val_train_ratio = 0.2

    """
    #Training with word2vec
    x_train, y_train = load_data()
    print("training data:", x_train.shape, y_train.shape)

    model = build_model((1, x_train.shape[2], x_train.shape[3]), conv_window_size, num_filters, reg, dropout)
    history = train(model, x_train, y_train, val_train_ratio, epochs, batch_size)

    print("Saving model...")
    model.model.save('model.h5')

    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, epochs+1), history.history['val_mean_absolute_error'], 'tab:blue', label="validation MAE")
    ax1.plot(range(1, epochs+1), history.history['mean_absolute_error'], 'tab:red', label="training MAE")

    ax2.plot(range(1, epochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()

    f.savefig('training.png', dpi=300)
    plt.show()
    print("Done.")

    """
    
    
    #Training with random init
    data = np.array([[0,"This is a sentence for doc0.",0.9], [1,"That is a sentence for doc1!" , 0.8], [2," And this is a sentence for doc2.", 0.6]])
    x_train, y_train = dummy_load_data(data)
    print("training data:", x_train.shape, y_train.shape)

    vocab_size = np.max(x_train)
    seq_len = len(x_train[0])
    model = build_model((1,1,1), conv_window_size, num_filters, reg, dropout, word2vec = False, max_token = vocab_size, sequence_len = seq_len)
    history = train(model, x_train, y_train, val_train_ratio, epochs, batch_size)

    print("Saving model...")
    model.model.save('model.h5')

    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, epochs+1), history.history['val_mean_absolute_error'], 'tab:blue', label="validation MAE")
    ax1.plot(range(1, epochs+1), history.history['mean_absolute_error'], 'tab:red', label="training MAE")

    ax2.plot(range(1, epochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()

    f.savefig('training.png', dpi=300)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
