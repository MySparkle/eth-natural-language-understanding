from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gensim import models

def construct_wordToID(train_file):
    """
        Args:
            train_file:     String, filename of the training data

        Returns:
            wordToID:       String to int, dictionary of the model
    """
    print("Constructing wordToID dictionary...")
    file = open(train_file, 'r')
    word_frequencies = {}
    i = 0
    for line in file:
        if i % 100000 == 0:
            print("at line:", i)
        for word in line.split():
            if word in word_frequencies.keys():
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1
        i += 1
    file.close()
    # Sort words with respect to frequency to pick the 20k most frequent words including the symbols
    num_most_frequent_words = 20000 - 4
    word_frequencies = dict(sorted(word_frequencies.items(), key=lambda x:x[1], reverse=True)[:num_most_frequent_words])
    wordToID = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}
    ID = 4
    for word in word_frequencies.keys():
        wordToID[word] = ID
        ID += 1
    return wordToID

def preprocess(file_name, wordToID):
    """
        Args:
            file_name:      String, filename
            wordToID:       String to int, dictionary of the model

        Returns:
            data:           numpy matrix of size (number of sentences in filename, 30), each word is substituted by their IDs in the dictionary
    """
    data = []
    print("Preprocessing sentences from {}...".format(file_name))
    file = open(file_name, 'r')
    i = 0
    long = 0
    for line in file:
        if i % 100000 == 0:
            print("at line:", i)
        words = line.split()
        # Ignore sentences more than 30 words long (including <bos> and <eos> tags)
        if len(words) > 28:
            long += 1
            continue
        data.append(np.arange(30))
        data[i][0] = wordToID["<bos>"]
        j = 1
        for word in words:
            if j > 28:
                break
            if word in wordToID.keys():
                data[i][j] = wordToID[word]
            else:
                data[i][j] = wordToID["<unk>"]
            j += 1
        data[i][j] = wordToID["<eos>"]
        j += 1
        # Add padding to sentences shorter than 30 words (including <bos> and <eos> tags)
        while j < 30:
            data[i][j] = wordToID["<pad>"]
            j += 1
        i += 1
    file.close()
    data = np.array(data)
    print("Sentences longer than 28 words: {}".format(long))
    return data


def load_data(file_name, wordToID):
    """
    Args:
        file_name:      String, filename
        wordToID:       String to int, dictionary of the model

    Returns:
        (x,y) where
        x:              inputs for the model
        y:              targets for the model
	"""
    data = preprocess(file_name, wordToID)
    x = data[:,:-1].astype(np.int32)
    y = data[:,1:].astype(np.int32)
    return (x, y)

def load_embeddings(session, wordToID, embeddings, embeddings_file, embeddings_size):
    """
    Args:
        session             Tensorflow session object
        wordToID            A dictionary mapping words to IDs
        embeddings          Embeddings tensor of shape vocabulary_size x embeddings_size
        embeddings_file     Path to the embeddings file
        embeddings_size     Dimensionality of the external embeddings
    """
    print("Loading external embeddings from {}...".format(embeddings_file))
    model = models.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    external_embeddings = np.zeros(shape=(len(wordToID), embeddings_size))
    matches = 0
    for word, ID in wordToID.items():
        if word in model.vocab:
            external_embeddings[ID] = model[word]
            matches += 1
        else:
            print("{} not in embeddings file".format(word))
            external_embeddings[ID] = np.random.uniform(low=-0.25, high=0.25, 
                                                        size=embeddings_size)
    print("{} words out of {} could be loaded".format(matches, len(wordToID)))
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = embeddings.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embeddings})
