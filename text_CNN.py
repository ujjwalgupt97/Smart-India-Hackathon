import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array, asarray, zeros
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pickle
import keras
from keras import backend
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
import os

import nltk

#Ignoring the warnings
import warnings
warnings.filterwarnings(action = 'ignore')

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.layers import LeakyReLU
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from keras.layers import LeakyReLU



def replace_contractions(sentence):
    """Replace contractions in string of text"""
    return contractions.fix(sentence)

  
def words_list(sample):
    words = []
    """Tokenising the corpus"""
    for i in sample: 
        temp = []
        for j in word_tokenize(i):
            temp.append(j.lower())
        temp = normalize(temp)
        words.append(temp)
    return words

  
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

  
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

  
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def normalize(words):
    """This is the main function which takes all other functions to pre-process the data given"""
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    lemmatize_words(words)
    return words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

  
def lemmatize_words(words):
    """Lemmatize the words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

#Dictionary of Class names and their index as in labels
label_name = {0: 'Acupuncture,Divine Care-9431757875,Accupress Healthcare Ranchi-06516008001', 
              1: 'Chiropractic,Shri Hari Spine Car-7725484379,Abhishek Health Care-9122352234',
              2: 'Diagnostic Lab,Diagno Labs-06572421212,Dr. Lal Path Labs Limited-09334807886,SRL Diagnostics-06572222228,Om Pathology Laboratory and Diagnostics-09431748841',
              3: 'Emergency Services,111 Hospital-08210102212,Meditrania Hospital-04742721111,Apex Hospital-06572432888,Meherbai Tata Memorial Hospital-06576641012',
              4: 'Human Resource Contact,Tata Memorial Hospital-06576641012,National Insurance-0657223132,Steel City Hospital-06572441724',
              5: 'Maternity,Dr. Mrs. Raghumoni-8235072251, Dr. Asha Gupta-9835373582, Dr. Mrs. Pinky Roy-8092010573',
              6: 'MRI-CAT Scan,Ideal Imaging Diagnostic Centre-06572421888,Dr. Kapoors Clinic-9334800433',
              7: 'Pharmacy Benefit Manager,REMEDI Pharma Jamshedpur-06572221888,Tata Health Medical Store-06572230026,Manish Medical-06572249193',
              8: 'Physician Visit Office Sick,Dr. S Kumar-8986709560,Dr. P Barnwal-9155704530,Dr. SR Prasad-9279374562',
              9: 'Psychiatric Out-patient,Dr. Sanjay Kumar-8092092060,Sumita Bhagat-9934506045,Mahesh Hembram-9204377421'}

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

#tf_config = some_custom_config
#sess = tf.Session(config=tf_config)
#graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
#set_session(sess)

#textcnn = load_model("./models/Text_CNN.h5")
#with open("./pickles/Text_CNN.pickle", 'rb') as handle:
#    textcnn_pickle = pickle.load(handle)

def predict_helper(sentence):
	#global textcnn,graph
	textcnn = load_model("./models/Text_CNN.h5")
	with open("./pickles/Text_CNN.pickle", 'rb') as handle:
	    textcnn_pickle = pickle.load(handle)
	word_list = words_list(sentence)
	X_test = []
	for list_of_words in word_list:
	    sentence = ' '.join(x for x in list_of_words)
	    X_test.append(sentence)
	encoded_word_list = textcnn_pickle.texts_to_sequences(X_test)
	X = pad_sequences(encoded_word_list, maxlen=15, padding='pre')
	print(X)
	#textcnn._make_predict_function()
	#global sess
	#global graph
	#with graph.as_default():
		#set_session(sess)
	pred = textcnn.predict(X)
	pred = np.argmax(pred)
	return label_name[pred]

#print(predict_helper(['I am Having back pain']))


