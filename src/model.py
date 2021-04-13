
import nltk
import json
import pickle

import numpy as np
from h5py.utils import check_numpy_read
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import random

from nltk.stem import WordNetLemmatizer

import tensorflow as tf

import requests
from datetime import datetime, timedelta

API_URL = "https://cricapi.com/api/"
API_KEY = "o1kw33cvFcgkgXyV2TyeMhhlw9p2"
# Creating GUI with tkinter
import tkinter
from tkinter import *

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

docs_folder = 'docs_folder/'

#words = []
#classes = []
#documents = []
#ignore_words = ['?', '!']
#data_file = open(docs_folder + 'management_intents.json').read()
#intents = json.loads(data_file)


def create_model():
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']
    docs_folder = 'docs_folder/'
    data_file = open(docs_folder + 'management_intents.json').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:

            # take each word and tokenize it
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # adding documents
            documents.append((w, intent['tag']))

            # adding classes to our class list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")

    print(len(classes), "classes", classes)

    print(len(words), "unique lemmatized words", words)

    pickle.dump(words, open(docs_folder + 'management_words.pkl', 'wb'))
    pickle.dump(classes, open(docs_folder + 'management_classes.pkl', 'wb'))

    # initializing training data
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        # initializing bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print("Training data created")

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # fitting and saving the model
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    model.save(docs_folder + 'chatbot_management_model.h5', hist)

    print("model created")
    return model
