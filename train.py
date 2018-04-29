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
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras.optimizers import Adadelta

from sklearn.utils import shuffle

# Build model
def build_model(input_shape, conv_window_size, num_filters, reg, dropout):
    model = Sequential()
    #model.add(Embedding(max_features,300))

    # we add a Convolution 1D, which will learn num_filters
    # word group filters of size conv_window_size:
    model.add(Conv2D(input_shape=input_shape,
                        filters=num_filters,
                        kernel_size=conv_window_size,
                        padding="valid",
                        activation="relu",
                        strides=1,
                        data_format='channels_first'))
    
    model.add(MaxPooling2D(pool_size=(num_filters, 1)))    
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='softmax', kernel_regularizer=regularizers.l2(reg)))
    
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
    data1 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency1.pickle", "rb"))
    data2 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency2.pickle", "rb"))
    data3 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency3.pickle", "rb"))
    data4 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency4.pickle", "rb"))
    data5 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency5.pickle", "rb"))
    data6 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency6.pickle", "rb"))
    data7 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency7.pickle", "rb"))
    data8 = pickle.load(open("/global/scratch/alex_vlissidis/wordEmbeddingsToSaliency8.pickle", "rb"))

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

def main():
    # Model Hyperparameters
    conv_window_size = (3, 300)
    num_filters = 400
    reg = 0.01
    dropout = 0.5
    
    # Training parameters
    epochs = 10
    batch_size = 256
    test_train_ratio = 0.2
    val_train_ratio = 0.2

    x_train, y_train = load_data()
    print("training data:", x_train.shape, y_train.shape)

    model = build_model((1, x_train.shape[2], x_train.shape[3]), conv_window_size, num_filters, reg, dropout)
    history = train(model, x_train, y_train, val_train_ratio, epochs, batch_size)

    print("Saving model...")
    model.model.save('model-softmax.h5')

    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, epochs+1), history.history['val_mean_absolute_error'], 'tab:blue', label="validation MAE")
    ax1.plot(range(1, epochs+1), history.history['mean_absolute_error'], 'tab:red', label="training MAE")

    ax2.plot(range(1, epochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()

    f.savefig('training-softmax.png', dpi=300)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
