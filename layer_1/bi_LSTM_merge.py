#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Build Character-enhanced token embedding layer to encode token.
# Step: 1. Transfer character into 37 binary vector.
#       2. Build bidirectional RNN layer with merge.
#             Input: character(37 binary vector) sequence of a token.
#             Output: ? dimension.
#          Train step:
#                      x: character(37 binary vector) sequence of a token.
#                      y: token correspond to GloVe result.
#       3. Concentrate GloVe result and lstm result. If no GloVe result, use lstm result instant of.

import logging
import pickle as pickle
import numpy as np
np.random.seed(19870712)  # for reproducibility
path = "../1_data/"

import nltk
import re
import h5py # It needs at save keras model
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.engine.topology import Merge
from scipy import spatial


chars= [' ', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
        'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def readfile(file_id = 2):

    if file_id == 1:
        # Read and arrange data set into x, y type.
        text_file = open(path + "glove.6B.100d.txt", 'r')
        glove_text = text_file.readlines()

        word_lens = 400000
        char_X_100 = 399488  # words: 399488
        char_X_010 = 43  # max word length: 43
        char_X_001 = 37  # chars: 37
        char_Y_10 = 399488  # words: 399488
        char_Y_01 = 100  # encode word length: 100

    if file_id == 2:
        # Read and arrange data set into x, y type.
        text_file = open(path + "glove.twitter.27B.100d.txt", 'r')
        glove_text = text_file.readlines()

        word_lens = 1193514
        char_X_100 = 695677  # words: 695677
        char_X_010 = 140  # max word length: 140
        char_X_001 = 37  # chars: 37
        char_Y_10 = 695677  # words: 695677
        char_Y_01 = 100  # encode word length: 100

        word_lens = 1193514
        char_X_100 = 695677  # words: 695677
        char_X_010 = 140  # max word length: 140
        char_X_001 = 37  # chars: 37
        char_Y_10 = 695677  # words: 695677
        char_Y_01 = 100  # encode word length: 100

    if file_id == 3:
        # Read and arrange data set into x, y type.
        text_file = open(path + "glove.840B.300d.txt", 'r')
        glove_text = text_file.readlines()

        word_lens = 2196017
        char_X_100 = 2193429  # words: 695677
        char_X_010 = 60  # max word length: >1000
        char_X_001 = 37  # chars: 37
        char_Y_10 = 2193429  # words: 695677
        char_Y_01 = 300  # encode word length: 100

    return [glove_text, [char_X_100, char_X_010, char_X_001, char_Y_10, char_Y_01], [word_lens]]


def what_d(runtimes=1, renew=True, maxlen=100, file_id=3):

    [glove,[char_X_100, char_X_010, char_X_001, char_Y_10, char_Y_01],[word_lens]] = readfile(file_id=file_id)

    char_X_010 = min(char_X_010, maxlen)

    vocab = []
    X = np.zeros((char_X_100, char_X_010, char_X_001), dtype=np.bool)
    y = np.zeros((char_Y_10, char_Y_01 ), dtype=np.float64)

    ii = 0
    for i in range(0, word_lens):
        ttt = glove[i].split()
        ttt_lens = len(ttt)
        lists = ["".join(ttt[0:ttt_lens - char_Y_01])] + ttt[ttt_lens - char_Y_01:]
        lists[0] = re.sub("[^0-9a-zA-Z]", "", lists[0].lower())
        if 0 < len(lists[0]) <= maxlen:
            #print(ii, i)
            vocab.append(lists[0])
            text = lists[0].ljust(char_X_010)
            for j in range(0, char_X_010):
                X[ii, j, char_indices[text[j]]] = 1
            for k in range(1, char_Y_01 + 1):
                y[ii, k - 1] = lists[k]
            ii = ii + 1
            if i % 40000 == 0:
                print(i)

    # Find par.
    lens = []
    for word in vocab:
        lens.append(len(word))
    print(max(lens))   # min(maxlen, char_X_010)
    print(len(vocab))  # 399488
    char_X_100 = len(vocab)
    char_Y_10 = len(vocab)
    X = X[0:len(vocab)]
    y = y[0:len(vocab)]


    # First time: build the model: a bidirectional SimpleRNN
    if renew == True:
        print('Build model...')
        left = Sequential()
        left.add(LSTM(char_Y_01, input_shape=(char_X_010, char_X_001), activation='tanh',
                           inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5))
                           #dropout_W=0.5, dropout_U=0.5))
        right = Sequential()
        right.add(LSTM(char_Y_01, input_shape=(char_X_010, char_X_001), activation='tanh',
                            inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5, go_backwards=True))
                            #dropout_W=0.5, dropout_U=0.5, go_backwards=True))
        model = Sequential()
        model.add(Merge([left, right], mode='sum'))
        model.compile('Adadelta', 'MSE', metrics=['accuracy'])
        model.fit([X, X], y, batch_size=512, nb_epoch=1)
        model.save(path + "layer_1/bi_LSTM_merge_" + str(file_id) + ".pk")


    # Not first time: build the model: a bidirectional LSTM

    print('Load model...')
    model = load_model(path+"layer_1/bi_LSTM_merge_" + str(file_id) + ".pk")
    for j in range(0,runtimes-1):
        print('Build model...')
        model.fit([X,X], y,
                  batch_size=512,
                  nb_epoch=1)
        model.save(path + "layer_1/bi_LSTM_merge_" + str(file_id) + ".pk")


    # Test cosine similarity, train set

    print('Test cosine similarity, train set')
    cos = []
    for i in range(0, len(vocab)):
        text = vocab[i].ljust(char_X_010)
        x = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)
        for j in range(0, len(text)):
            x[0, j, char_indices[text[j]]] = 1
        map_LSTM = model.predict([x, x], verbose=0)

        map_GloVe = y[i]

        cos.append(1 - spatial.distance.cosine(map_LSTM, map_GloVe))
    f = open(path+"layer_1/cosine.txt", 'a')
    f.write("20 times bi_LSTM_merge" + str(file_id) + " cosine similarity: "+str(sum(cos)/len(cos))+"\n")
    f.close()


    # Test cosine similarity, misspelling

    print('Test cosine similarity, misspelling')
    cos = []
    change_engs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in range(0, len(vocab)):
        misspelling = vocab[i]
        if len(misspelling)>4:
            loc = int(np.random.uniform(0,1,1)*len(misspelling))
            cha = int(np.random.uniform(0,1,1)*26)

            tem = list(misspelling)
            tem[loc] = change_engs[cha]

            misspelling = "".join(tem)
            text = misspelling.ljust(char_X_010)
            x = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)
            for j in range(0, len(text)):
                x[0, j, char_indices[text[j]]] = 1
            map_LSTM = model.predict([x, x], verbose=0)
            map_GloVe = y[i]

            cos.append(1 - spatial.distance.cosine(map_LSTM, map_GloVe))
    f = open(path+"layer_1/cosine.txt", 'a')
    f.write("20 times bi_LSTM_merge" + str(file_id) + " misspelling cosine similarity : "+str(sum(cos)/len(cos))+", len: "+str(len(cos))+"\n")
    f.close()

what_d(runtimes =  20, renew =True,  maxlen = 18, file_id=3)

print ("end")
