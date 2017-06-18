# ETH Natural Language Understanding Project (Spring 2017)
## Task 1: RNN Language Modelling
In the first part of this task we build a simple LSTM language model in Tensorflow, constructing the actual RNN by unrolling the graph. We experiment with pretrained word embeddings and a layer that down-projects the hidden state before the softmax.
In the second part of this task we use the trained language model to generate sentences in a greedy fashion.

## Task 2: Dialogue
In the second task we build a dialogue system. To develop our model, we use Google's tf-seq2seq
library for Tensorflow, implementing seq2seq architectures.
As our baseline, we use a basic seq2seq model with bidirectional RNN encoder, whose last state is passed as the initial state of the basic decoder. However, given the poor results of the baseline, we try many approaches to improve our model.
Firstly, we increase the training data by 50% by merging the Cornell Movie-Dialogs Corpus with the given MovieTriples dataset. In addition, we try the Bahdanau attention and Dot attention mechanisms. Moreover, we experiment with residual connections on the encoder and decoder, as well as using a convolutional encoder. Finally, we improve the results of our best model by exploiting the three-turn structure of the dataset and increasing the size and depth of the model.
