# 66.80 percent accuracy, 0.7837 loss, achieves this fast then over fits

import numpy as np
import string
import pandas as pd
import nltk
nltk.download("popular")
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, CuDNNLSTM, Conv1D, MaxPool1D, BatchNormalization
from keras.optimizers import SGD
from keras import metrics

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
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
            word_idx = vocab_dict.get(word, 1)  # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name, engine="python", error_bad_lines=False)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0  # 0 means the padding signal
    vocab_dict['<unk>'] = 1  # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    data_matrix = get_sequence(df['words'], input_length, vocab_dict)
    stars = df['stars'].apply(int) - 1
    return df['review_id'], stars, data_matrix, vocab


# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("train.csv", input_length)
    K = max(train_data_label) + 1  # labels begin with 0

    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, vocab = read_data("valid.csv", input_length, vocab=vocab)

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("test.csv", input_length, vocab=vocab)

    print("Vocabulary Size:", len(vocab))
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
           test_id_list, test_data_matrix, None, vocab


if __name__ == '__main__':
    # Hyperparameters
    input_length = 300
    embedding_size = 100
    hidden_size = 100
    batch_size = 100
    conv_filters = 64
    dropout_input = 0.5
    dropout_hidden = 0.6
    learning_rate = 0.1
    total_epoch = 30

    train_id_list, train_data_matrix, train_data_label, \
    valid_id_list, valid_data_matrix, valid_data_label, \
    test_id_list, test_data_matrix, _, vocab = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K

    # New model
    model = Sequential()

    # embedding layer and dropout
    # YOUR CODE HERE
    model.add(Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_length))
    model.add(Dropout(dropout_input))

    model.add(Conv1D(filters=conv_filters, kernel_size=5, activation='relu'))
    # model.add(MaxPool1D(pool_size=3))
    model.add(BatchNormalization())

    # LSTM layer
    # YOUR CODE HERE
    model.add(Bidirectional(CuDNNLSTM(units=hidden_size)))
    model.add(Dropout(dropout_hidden))

    # output layer
    # YOUR CODE HERE
    model.add(Dense(K, activation='softmax'))

    # SGD optimizer with momentum
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # training
    model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size, validation_data=(valid_data_matrix, valid_data_label))
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)
