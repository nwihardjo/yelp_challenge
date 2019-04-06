# -*- coding: utf-8 -*-
import numpy as np
import string
import os
import pandas as pd
import nltk
import tensorflow as tf
from symspellpy.symspellpy import SymSpell
from textblob import TextBlob
import matplotlib.pyplot as plt

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Bidirectional,\
#     CuDNNLSTM, Conv1D, GlobalMaxPooling1D, Reshape, Permute, Lambda, concatenate, CuDNNGRU, *
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import metrics

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english') + list(string.punctuation))

def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens



def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab, train_data_matrix2 = read_data("./data/train_big.csv", input_length)
    K = max(train_data_label)+1  # labels begin with 0

    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, vocab, valid_data_matrix2 = read_data("./data/valid_v5.csv", input_length, vocab=vocab)

    # Load testing data
    test_id_list, _, test_data_matrix, _, test_data_matrix2 = read_data("./data/test_v5.csv", input_length, vocab=vocab)
    
#     print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Validation Set Size:", len(valid_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Validation Set Shape:", valid_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=K)
    return train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, None, vocab, train_data_matrix2, valid_data_matrix2, \
        test_data_matrix2

#     print ("vocabulary size: ", len(vocab))
    print ("training set size: ", len(train_id_list))
    print("validation set size: ", len (valid))

def get_dict(vocab):
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0 # 0 means the padding signal
    vocab_dict['<unk>'] = 1 # 1 means the unknown word
    vocab_size = 2
    
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1
        
    return vocab_dict

    
def train_embed(mode, vocab, embed_dim=100):
    # create dict of word as the key and the pre-trained weights as the value
    print('Getting and preparting pre-trained matrix...')
    embeddings_index = dict()
    
    if mode == 'glove':
        if embed_dim == 100:
            fname = './data/glove.6B.100d.txt'
        else:
            fname = './data/glove.6B.300d.txt'

        f = open(fname)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(vocab)+2, embed_dim), dtype=int)
        vocab_dict = get_dict(vocab)

        for word, i in vocab_dict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
    elif mode == 'stanford_1D':
        fname = 'dictionary.txt'

def get_sequence(data, seq_length, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    '''
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # YOUR CODE HERE
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1) # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix

def textblob_analyse(review):
    vect1 = np.vectorize(lambda obj: TextBlob(obj).sentiment.polarity)    
    vect2 = np.vectorize(lambda obj: TedefxtBlob(obj).sentiment.subjectivity)
    
    return np.column_stack((vect1(review), vect2(review)))

def clean_text(text):
    text = ''.join(tokenize(text))
    
    max_edit_distance_dictionary = 2
    prefix_length = 7
    
    ss = SymSpell(max_edit_distance_dictionary, prefix_length)
    dict_path = "./frequency_dictionary_en_82_765_GLOVE.txt"
    if not ss.load_dictionary(dict_path, 0, 1):
        print('DEBUG: DICTIONARY FILE NOT FOUND!!!')
            
    suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit_distance_dictionary))[0].term
    
    return suggestion.split()
    

def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    print('Reading data from ', file_name, '...')
    df = pd.read_csv(file_name)
    print('\t Cleaning text from', file_name, '...')
    df['words'] = df['text'].apply(tokenize)
    
    if vocab is None:
        print('\t Generating vocab...')
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = get_dict(vocab)

    print('\t Getting sequence of text from ', file_name, '...')
    data_matrix = get_sequence(df['words'], input_length, vocab_dict)
    
    print('\t Normalising additional matrix...')
    data_matrix2 = normalize(np.column_stack((df['cool'], df['funny'],df['useful'])))
    
    #print('\t Analysing text blob from ', file_name, '...')
    #data_matrix2 = np.column_stack((data_matrix2, textblob_analyse(df['text'])))
    
    stars = df['stars'].apply(int) - 1
    return df['review_id'], stars, data_matrix, vocab, data_matrix2


if __name__ == '__main__':
    # Hyperparameters
    input_length = 150
    embedding_size = 300
    hidden_size = 100
    batch_size = 160
    dropout_rate = 0.5
    learning_rate = 0.01
    total_epoch = 15

    train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, _, vocab, train_matrix_add, \
        valid_matrix_add, test_matrix_add = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K
    
    pretrain_matrix=train_embed('glove', vocab, embedding_size)
    
    
    embed_in = Input(shape = (input_length,), dtype='int32', name='embed_in')
    x = Embedding(input_dim = input_size, output_dim = embedding_size, \
                  input_length=input_length, weights=[pretrain_matrix])(embed_in)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=hidden_size)) (x)  
    lstm_out = Dropout(dropout_rate)(x)
    
    aux_in = Input(shape=(3,), name='aux_in')
    dense_in = concatenate([lstm_out, aux_in])
    
    x1 = Dense(hidden_size, activation='linear')(dense_in)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Dense(hidden_size, activation='linear')(x1)
    pred = Dense(output_size, activation='softmax')(x1)
    
    optimizer = RMSprop(lr = learning_rate)
    
    model = Model(inputs=[embed_in, aux_in], outputs=pred)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    history = model.fit([train_data_matrix, train_matrix_add], train_data_label, 
          validation_data = ([valid_data_matrix, valid_matrix_add], valid_data_label),
          epochs=total_epoch, batch_size=batch_size)

    train_score = model.evaluate([train_data_matrix, train_matrix_add], train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = model.evaluate([valid_data_matrix, valid_matrix_add], valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = model.predict([test_data_matrix, test_matrix_add], batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)

print(history.history.keys())

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')
