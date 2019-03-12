import numpy as np
import string
import pandas as pd
import nltk
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, Conv2D, MaxPool2D, Concatenate, Input, Reshape, Flatten
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
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1) # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0 # 0 means the padding signal
    vocab_dict['<unk>'] = 1 # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    data_matrix = get_sequence(df['words'], input_length, vocab_dict)

    return df['id'], df['label']-1, data_matrix, vocab
# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv", input_length)
    K = max(train_data_label)+1  # labels begin with 0

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", input_length, vocab=vocab)
    test_data_label = pd.read_csv("data/answer.csv")['label'] - 1
    
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    test_data_label = keras.utils.to_categorical(test_data_label, num_classes=K)
    return train_data_matrix, train_data_label, test_data_matrix, test_data_label, vocab


if __name__ == '__main__':
    # Hyperparameters
    input_length = 30
    embedding_size = 100
    hidden_size = 100
    batch_size = 100
    dropout_rate = 0.5
    filters = 100
    kernel_sizes = [3, 4, 5]
    padding = 'valid'
    activation = 'relu'
    strides = 1
    pool_size = 2
    learning_rate = 0.1
    total_epoch = 10

    train_data_matrix, train_data_label, test_data_matrix, test_data_label, vocab = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K

    # New model
    # YOUR CODE HERE
    x = None

    # embedding layer and dropout
    # YOUR CODE HERE
    e = None
    e_d = None

    # construct the sequence tensor for CNN
    # YOUR CODE HERE
    e_d = None

    # CNN layers
    conv_blocks = []
    for kernel_size in kernel_sizes:
        # YOUR CODE HERE
        conv = None
        maxpooling = None
        faltten = None
        conv_blocks.append(faltten)

    # concatenate CNN results
    # YOUR CODE HERE
    c = None
    c_d = None

    # dense layer
    # YOUR CODE HERE
    d = None

    # output layer
    # YOUR CODE HERE
    y = None

    # build your own model
    # YOUR CODE HERE
    model = None

    # SGD optimizer with momentum
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # training
    model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    test_score = model.evaluate(test_data_matrix, test_data_label, batch_size=batch_size)

    print('Training Loss: {}\n Training Accuracy: {}\n'
          'Testng Loss: {}\n Testing accuracy: {}'.format(
              train_score[0], train_score[1],
              test_score[0], test_score[1]))
