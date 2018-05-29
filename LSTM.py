from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from numpy import array
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# Understood concept and code from https://github.com/EmielStoelinga/CCMLWI 

MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 20000
EMBED_SIZE = 100
BATCH_SIZE = 256
NUM_EPOCHS = 20
NUM_FILTERS = 32
NUM_WORDS = 3

class LSTMClassifier:
    def __init__(self, companyName, inputFilename, NumberOfDays):
        self.companyName = companyName
        self.NumberOfDays = NumberOfDays
        self.inputFilename = inputFilename
    
    def performLstm(self):
        print("reading data...")
        df=pd.read_csv(self.inputFilename)
        df = df[df['Company'] == self.companyName]

        counter = collections.Counter()
        text = df.Tweet
        maxlen = 0
        for line in text:
            sent = line.strip()
            words = [x.lower() for x in nltk.word_tokenize(sent)]
            if len(words) > maxlen:
                maxlen = len(words)

        text = df.Tweet
        tokenizer = Tokenizer(VOCAB_SIZE)
        tokenizer.fit_on_texts(text)
        all_x = tokenizer.texts_to_sequences(text)

        W = pad_sequences(all_x, maxlen=maxlen)
        sentiment = df.Sentiment.values.reshape((df.Sentiment.shape[0], 1))
        W = np.append(W, sentiment, 1)
        maxlen = maxlen + 1

        sp_5 = df['T-5'].values.reshape((df['T-5'].shape[0], 1))
        W = np.append(W, sp_5, 1)
        maxlen = maxlen + 1

        sp_4 = df['T-4'].values.reshape((df['T-4'].shape[0], 1))
        W = np.append(W, sp_4, 1)
        maxlen = maxlen + 1

        sp_3 = df['T-3'].values.reshape((df['T-3'].shape[0], 1))
        W = np.append(W, sp_3, 1)
        maxlen = maxlen + 1

        sp_2 = df['T-2'].values.reshape((df['T-2'].shape[0], 1))
        W = np.append(W, sp_2, 1)
        maxlen = maxlen + 1

        sp_1 = df['T-1'].values.reshape((df['T-1'].shape[0], 1))
        W = np.append(W, sp_1, 1)
        maxlen = maxlen + 1

        print("Converting labels to One Hot Vectors...")
        predictForLabel = 'T+{0}'.format(self.NumberOfDays)   # T+1 for tomorrow, T+2 is day after tomorrow
        Y = df.as_matrix(columns=[predictForLabel]).flatten()


        print("Train-Test split")
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(W, Y, test_size=0.3, random_state=42)
        Xtrain, XValidate, Ytrain, YValidate = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=42)
        print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)


        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen, trainable=True))
        model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
        model.add(MaxPooling1D(pool_size=1))
        model.add(LSTM(maxlen))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(XValidate, YValidate))

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])    
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train set', 'test set'], loc='best')
        plt.title('Train vs Validation Accuracy')
        plt.savefig('TrainAndValidation.png')



        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('TrainAndValidationLoss.png')

        model.summary()

        y_pred = model.predict(Xtest)
        y_pred = np.rint(y_pred)
        y_pred = y_pred.astype(int)
        y_pred = y_pred.flatten()

        correct = 0
        for i in range(len(Ytest)):
            if Ytest[i] == y_pred[i]:
                correct += 1

        accuracy = 100.0*correct/len(Ytest)
        print ('For Company {0} and Today+{1} days, LSTM test accuracy is {2}'.format(self.companyName, self.NumberOfDays, accuracy))




