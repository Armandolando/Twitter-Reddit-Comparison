import time
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset
import pandas as pd
#from sentiment.load_datasets import load_emotion_tweets, load_binary_tweets, load_generic_tweets, load_elections_tweets
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Embedding, Input, Activation, concatenate, GlobalMaxPooling1D, \
    SpatialDropout1D, LSTM, MaxPooling3D, MaxPooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from numpy import asarray, zeros, array
from collections import  Counter
import numpy as np
#from sentiment import build_vocabulary, extract_features
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import random

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from nltk.stem import PorterStemmer
from nltk import RegexpTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import collections, csv, os
from stanfordcorenlp import StanfordCoreNLP
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def get_words_stop(data):
    """
    Reads the text passed as input, formatting the words in a more suitable structure (no punctuation,
    lower case); it is also applied the stopwords technique.
    :param data: list of texts
    :return: list of words in the text
    """
    #nlp = StanfordCoreNLP(r'/home/marco/projects/reddit_twitter_analysis/CoreNlp/stanford-corenlp-full-2018-10-05')
    
    #props={'annotators': 'tokenize,ssplit,pos,lemma','pipelineLanguage':'en','outputFormat':'json'}
    stemmer = PorterStemmer()
    words = []
    p = "|\"#%$()'+,-./;:<=>?`!@[]^_{}-"
    table = str.maketrans(p, " " * len(p))
    for comment in data:
        
        formatted_line = comment.translate(table).lower()
        for w in formatted_line.split():
            if len(w) > 2 and w.lower() not in stopwords.words('english'):
                words.append(stemmer.stem(w)) #stemming
        '''
        r = nlp.annotate(comment, properties=props)
        for sentence in json.loads(r)['sentences']:
            for token in sentence['tokens']:
                if not token['lemma'] in stopwords.words('english'):
                    words.append(token['lemma'])
        '''
    return words

def create_voc(data):
    """
    Uses the 'get_words' functions to format the text and builds the structure of the vocabulary
    :param train_file: file used to build the vocabulary
    :param stop: flag to enable stopping technique
    :param stem: flag to enable stemming technique
    :return: two vocabularies, one about the titles, the other about the publishers' names
    """
    voc = collections.Counter()
    words = get_words_stop(data)
    print(f'count {len(set(words))}')
    voc.update(words)
    return voc

def write_voc(voc, filename, n):
    """
    Saves the vocabulary as a txt file, keeping the n most frequent words
    :param voc: vocabulary to save
    :param filename: name to be assigned to the new file
    :param n: number of most frequent words to keep
    """
    f = open(filename, "w", encoding="utf8")
    for word, count in sorted(voc.most_common(n)):
        print(word, file=f)
    f.close()

def over_sampling(X, Y):
    oversample = RandomOverSampler(sampling_strategy='auto')
    Xover, Yover = oversample.fit_resample(X, Y)
    return Xover, Yover

def under_sampling(X, Y):
    undersample = RandomUnderSampler(sampling_strategy='auto')
    Xunder, Yunder = undersample.fit_resample(X, Y)
    return Xunder, Yunder

def clean_comments(comments, labels, vocab, stem=False):
    """
    Pre-process the comments, applying:
        - punctuation cleaning,
        - stemming
        - lower case
        - stopwords
        - vocabulary built
    :param comments: list of raw comments
    :param vocab: vocabulary containing the words to be kept
    :param stem: flag to control the application of the stemming
    :return: list of cleaned comments; each element of the list is a list of words
    """
    cleaned_c = []
    selected_labels = []
    tokenizer = RegexpTokenizer(r'\w+')
    #nlp = StanfordCoreNLP(r'/home/marco/projects/reddit_twitter_analysis/CoreNlp/stanford-corenlp-full-2018-10-05')
    
    #props={'annotators': 'tokenize,ssplit,pos,lemma','pipelineLanguage':'en','outputFormat':'json'}
    #f = open('stopwords.txt', encoding="utf8")
    #stopwords = f.read()
    #voc = vocab
    #f.close()
    stemmer = PorterStemmer()
    for comment, label, _ in zip(comments, labels, tqdm(range(len(comments)))):
        
        no_punct = tokenizer.tokenize(comment) # no punctuation
        if stem:
            lower_comment = [stemmer.stem(w.lower()) for w in no_punct]  # applying low case + stemming
        else:
            lower_comment = [w.lower() for w in no_punct] # applying low case
        words = [w for w in lower_comment if w not in stopwords.words('english')]  # applying stopwords
        tokens = [t for t in words if t in vocab] # filter by vocabulary
        #print(tokens)
        cleaned_c.append(tokens)
        selected_labels.append(label)
        '''
        new_text = []
        r = nlp.annotate(comment, properties=props)
        for sentence in json.loads(r)['sentences']:
            for token in sentence['tokens']:
                if not token['lemma'] in stopwords.words('english') and token['lemma'] in vocab:
                    new_text.append(token['lemma'])
        #print(new_text)
        cleaned_c.append(new_text)
        selected_labels.append(label)
        
    nlp.close()
    '''
    return cleaned_c, selected_labels

def load_embedding(filename):
    """
    Loads a word2vec structure already saved
    :param filename: path of the word2vec file
    :return: dictionary of the word2vec embedding:
                 - key:   word
                 - value: array of floats, representing the associated word embedding
    """
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding

def get_weight_matrix(embedding, vocab, dimension):
    """
    Puts in a matrix the embeddings for each word in the corpus (present in the vocabulary), in the right order
    :param embedding: output of the 'load_embedding()' function
    :param vocab: dictionary with the following structure:
                      - key:   word
                      - value: index reflecting a rank of frequencies (e.g.: 1 -> most frequent word)
    :return: matrix of word embeddings
    """
    # total vocabulary size plus 1 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, dimension))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix

def naive_bayes(comments, labels, vocab, stem, n_words, mode, balance):
    """
    Performs Naive Bayes
    :param balanced_c: list of comments
    :param balanced_l: list of associated labels
    :param stem: flag that controls the application or not of stemming
    :param n_words: number of words composing the vocabulary
    :param mode: accepts two different string values:
                     - 'bow': feature extraction is based on BoW
                     - 'tf-idf': feature extraction is based on TF-IDF
    :param balance: accepts two different string values:
                        - 'under': applies under-sampling
                        - 'over': applies over-sampling
    """
    comments, _ = clean_comments(comments, labels, vocab, stem)
    comments = [" ".join(comment) for comment in comments]
    if mode == 'bow':
        print('\nComputing BoW...') # provare tf-idf, media vettori parole word2vec
        extract_features.main(comments, labels, stem)
        print('BoW computed')
        data = np.loadtxt('dataset.txt.gz')
        X = data[:, :n_words]
        Y = np.ravel(data[:, n_words:])
    if mode == 'tf-idf':
        vectorizer = TfidfVectorizer(max_features=n_words)
        X = vectorizer.fit_transform(comments)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(array(labels))
        Y = integer_encoded

    if balance == 'under':
        X, Y = under_sampling(X, Y)
        print('\nClass distribution after under-sampling:')
        print(Counter(Y))
    if balance == 'over':
        X, Y = over_sampling(X, Y)
        print('\nClass distribution after over-sampling')
        print(Counter(Y))

    n_labels = len(Counter(labels).keys())
    if n_labels == 2:
        ticks = ['negative', 'positive']
    if n_labels == 3:
        ticks = ['negative', 'neutral', 'positive']
    if n_labels == 6:
        ticks = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    mnb = MultinomialNB()
    print('\nTraining model...')
    mnb.fit(X_train, Y_train)
    print('Model trained')
    print('\nTesting model')
    mean_acc = mnb.score(X_test, Y_test)
    print('Test accuracy: ', mean_acc)
    predictions = mnb.predict(X_test)
    print(classification_report(Y_test, predictions))
    conf_mat = confusion_matrix(Y_test, predictions, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, xticklabels=ticks,
                yticklabels=ticks, cmap="YlGnBu")
    title = 'Confusion matrix for MNB classifier\nMean test accuracy = ' + str(round(mean_acc*100, 2)) + '%'
    plt.title(title)
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.show()

def lstm(n_labels, embedding_layer, length):
    # define model
    # parameters of the output layer that must be changed according to the number of classes
    if n_labels == 2:
        activation_funct = 'sigmoid'
        loss_funct = 'binary_crossentropy'
        n_out = 1
    else:
        activation_funct = 'softmax'
        loss_funct = 'categorical_crossentropy'
        n_out = n_labels

    data_input = Input(shape=(length,))

    encoder = embedding_layer(data_input)
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(encoder)
    bigram_branch = MaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(encoder)
    trigram_branch = MaxPooling1D()(trigram_branch)
    merged = concatenate([bigram_branch, trigram_branch], axis=1)

    merged = LSTM(64, dropout=0.4, recurrent_dropout=0.4)(merged)

    merged = Dense(64, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.2)(merged)
    merged = Dense(n_out)(merged)
    output = Activation(activation_funct)(merged)
    model = Model(inputs=[data_input], outputs=[output])
    model.compile(loss=loss_funct, optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def parallel_cnn(n_labels, embedding_layer, length):
    """
    Implements a CNN with different branches working in parallel
    :param n_labels: number of output classes
    :param embedding_layer: instance of the Embedding class
    :param length: maximum lenght that can be assumed by a sentence
    :return: model implemented
    """
    # define model
    # parameters of the output layer that must be changed according to the number of classes
    if n_labels == 2:
        activation_funct = 'sigmoid'
        loss_funct = 'binary_crossentropy'
        n_out = 1
    else:
        activation_funct = 'softmax'
        loss_funct = 'categorical_crossentropy'
        n_out = n_labels

    data_input = Input(shape=(length,))

    encoder = embedding_layer(data_input)
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    merged = concatenate([bigram_branch, trigram_branch], axis=1)

    merged = Dense(256, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.75)(merged)
    merged = Dense(n_out)(merged)
    output = Activation(activation_funct)(merged)
    model = Model(inputs=[data_input], outputs=[output])
    model.compile(loss=loss_funct, optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def sequential_cnn(n_labels, embedding_layer):
    """
    Implements a CNN with sequential layers
    :param n_labels: number of output classes
    :param embedding_layer: instance of the Embedding class
    :return: model implemented
    """
    # define model
    # parameters of the output layer that must be changed according to the number of classes
    if n_labels == 2:
        activation_funct = 'sigmoid'
        loss_funct = 'binary_crossentropy'
        n_out = 1
    else:
        activation_funct = 'softmax'
        loss_funct = 'categorical_crossentropy'
        n_out = n_labels
    # provare diversi rami paralleli, attention layer...
    model = Sequential()

    model.add(embedding_layer)
    model.add(Conv1D(filters=100, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    #model.add(Conv1D(filters=100, kernel_size=2, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(n_out, activation=activation_funct))
    print(model.summary())
    # compile network
    model.compile(loss=loss_funct, optimizer='adam', metrics=['accuracy'])
    return model

def nn(balanced_c, balanced_l, emb_dimension, stem, balance):
    """
    Fine tunes a CNN, starting from Word2Vec weights in the first layer
    Assumes the dataset being already balanced
    :param balanced_c: list of comments
    :param balanced_l: list of associated labels
    :param emb_dimension: dimensionality of the word embeddings
    :param stem: flag that controls the application or not of stemming
    """
    f = open('vocabulary.txt', encoding='utf8')
    vocab = f.read()
    f.close()
    # load the vocabulary
    n_labels = 6
    balanced_c, l = clean_comments(balanced_c, balanced_l, vocab, stem)
    #print(balanced_c)
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(balanced_c)
    # encode all the comments
    encoded_docs = tokenizer.texts_to_sequences(balanced_c)
    # get max number of tokens in a comment
    max_length = max([len(s) for s in balanced_c])
    # zero padding, where the width is given by the document with most tokens
    encoded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    if balance == 'over':
        encoded_docs, balanced_l = over_sampling(encoded_docs, balanced_l)
    if balance == 'under':
        encoded_docs, balanced_l = under_sampling(encoded_docs, balanced_l)
    else:
        pass
    print(Counter(balanced_l))

    balanced_l = tf.keras.utils.to_categorical(balanced_l) # one-hot encoding
    

    X_train, Xtest, Y_train, ytest = train_test_split(encoded_docs, balanced_l, test_size=0.20, random_state=42)
    Xtrain, Xval, ytrain, yval = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    # train word2vec model
    model = Word2Vec(balanced_c, vector_size=emb_dimension, window=5, workers=8, min_count=1, sg=1)
    # summarize vocabulary size in model
    words = list(model.wv.index_to_key)
    print('Vocabulary size: %d' % len(words))

    # save model in ASCII (word2vec) format
    filename = 'embedding_word2vec.txt'
    model.wv.save_word2vec_format(filename, binary=False)

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    # load embedding from file
    raw_embedding = load_embedding('embedding_word2vec.txt')
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, emb_dimension)
    # create the embedding layer
    embedding_layer = Embedding(vocab_size, emb_dimension, weights=[embedding_vectors], input_length=max_length, trainable=True)

    #model = sequential_cnn(n_labels, embedding_layer) # function were the architecture is defined
    #model = parallel_cnn(n_labels, embedding_layer, max_length)
    model = lstm(n_labels, embedding_layer, max_length)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    # fit network
    history = model.fit(Xtrain, ytrain, batch_size=150, epochs=30, validation_data=(Xval, yval) ,verbose=2,
                        callbacks=[checkpointer, earlyStopping])

    plt.cla()
    plt.title('Accuracy')
    x_list = list(range(len(history.history['accuracy'])))
    plt.plot(x_list, history.history['accuracy'], 'b', label='Train')
    plt.plot(x_list, history.history['val_accuracy'], 'r', label='Validation')
    plt.legend()
    plt.grid()
    plt.show()

    # evaluate
    model.load_weights('model.weights.best.hdf5')
    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    probabilities = model.predict(Xtest)
    predictions = []
    if n_labels == 2:
        ticks = ['negative', 'positive']
        for sample in probabilities:
            if sample >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
    else:
        if n_labels == 3:
            ticks = ['negative', 'neutral','positive']
        else:
            ticks = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        for sample in probabilities:
            predictions.append(list(sample).index(max(list(sample))))
        ytest = np.argmax(ytest, axis=1)
    print(classification_report(ytest, predictions, digits=4))
    cm = confusion_matrix(ytest, predictions, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, xticklabels=ticks,
                yticklabels=ticks, cmap="YlGnBu")
    title = 'Confusion matrix for CNN with parallel branches\nMean test accuracy = ' + str(round(acc * 100, 2)) + '%'
    plt.title(title)
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.show()

    print('\nTest Accuracy: %f' % (acc * 100))

def k_fold_nn(balanced_c, balanced_l, emb_dimension, stem, balance):
    """
    Fine tunes a CNN, starting from Word2Vec weights in the first layer
    Assumes the dataset being already balanced
    :param balanced_c: list of comments
    :param balanced_l: list of associated labels
    :param emb_dimension: dimensionality of the word embeddings
    :param stem: flag that controls the application or not of stemming
    """
    f = open('vocabulary.txt', encoding='utf8')
    vocab = f.read()
    f.close()
    # load the vocabulary
    n_labels = 6
    balanced_c, l = clean_comments(balanced_c, balanced_l, vocab, stem)
    #print(balanced_c)
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(balanced_c)
    # encode all the comments
    encoded_docs = tokenizer.texts_to_sequences(balanced_c)
    # get max number of tokens in a comment
    max_length = max([len(s) for s in balanced_c])
    # zero padding, where the width is given by the document with most tokens
    encoded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    if balance == 'over':
        encoded_docs, balanced_l = over_sampling(encoded_docs, balanced_l)
    if balance == 'under':
        encoded_docs, balanced_l = under_sampling(encoded_docs, balanced_l)
    else:
        pass
    print(Counter(balanced_l))
    

    X_train, Xtest, Y_train, ytest = train_test_split(encoded_docs, balanced_l, test_size=0.20, random_state=42)
    #Xtrain, Xval, ytrain, yval = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    # train word2vec model
    model = Word2Vec(balanced_c, vector_size=emb_dimension, window=5, workers=8, min_count=1, sg=1)
    # summarize vocabulary size in model
    words = list(model.wv.index_to_key)
    print('Vocabulary size: %d' % len(words))

    # save model in ASCII (word2vec) format
    filename = 'embedding_word2vec.txt'
    model.wv.save_word2vec_format(filename, binary=False)

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    # load embedding from file
    k=0
    VALIDATION_ACCURACY = []
    VALIDATION_PRECISION = []
    VALIDATION_RECALL = []
    VALIDATION_LOSS = []

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    for train, val in kfold.split(X_train, Y_train):
        k+=1
        print(f"K = {k}")

        raw_embedding = load_embedding('embedding_word2vec.txt')
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, emb_dimension)
        # create the embedding layer
        embedding_layer = Embedding(vocab_size, emb_dimension, weights=[embedding_vectors], input_length=max_length, trainable=True)

        Ytrain = tf.keras.utils.to_categorical(Y_train) # one-hot encoding
        #model = sequential_cnn(n_labels, embedding_layer) # function were the architecture is defined
        #model = parallel_cnn(n_labels, embedding_layer, max_length)
        model = lstm(n_labels, embedding_layer, max_length)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
        # fit network
        model.fit(X_train[train], Ytrain[train], batch_size=150, epochs=20, validation_data=(X_train[val], Ytrain[val]) ,verbose=2,
                            callbacks=[earlyStopping])
        
        results = model.evaluate(X_train[val], Ytrain[val], verbose=2)
        #print(results)
        results = dict(zip(model.metrics_names,results))

        probabilities = model.predict(X_train[val])
        predictions = []
        if n_labels == 2:
            ticks = ['negative', 'positive']
            for sample in probabilities:
                if sample >= 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
        else:
            if n_labels == 3:
                ticks = ['negative', 'neutral','positive']
            else:
                ticks = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            for sample in probabilities:
                predictions.append(list(sample).index(max(list(sample))))
            yval = np.argmax(Ytrain[val], axis=1)

        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])
        VALIDATION_PRECISION.append(precision_score(yval, predictions, average='weighted'))
        VALIDATION_RECALL.append(recall_score(yval, predictions,average='weighted'))

        tf.keras.backend.clear_session()
    print(sum(VALIDATION_ACCURACY)/len(VALIDATION_ACCURACY))
    print(sum(VALIDATION_PRECISION)/len(VALIDATION_PRECISION))
    print(sum(VALIDATION_RECALL)/len(VALIDATION_RECALL))

    


emotion_dataset = load_dataset("emotion")

emotion_dataset.set_format(type="pandas")
train = emotion_dataset["train"][:]
test = emotion_dataset["test"][:]
val = emotion_dataset["validation"][:]

data = pd.concat([train, test, val])

voc = create_voc(data['text'])
write_voc(voc, 'vocabulary.txt', 6000)

nn(data['text'], data['label'], 100, True, balance='no')
#k_fold_nn(data['text'], data['label'], 100, True, balance='no')
#naive_bayes(data['text'], data['label'], voc, True, 1000, mode='tf-idf', balance='no')
