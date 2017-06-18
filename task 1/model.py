from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from os.path import join as path_join

from preprocess import construct_wordToID, load_data, load_embeddings
from rnn import static_rnn

class Trainer(object):
    def __init__(self, 
                  args, 
                  wordToID, 
                  timestep, 
                  classes, 
                  embeddings_size,
                  max_pred_sentence_len,
                  scope=None):
        self._args = args
        self._wordToID = wordToID
        self._timestep = timestep
        self._classes = classes
        self._emd_size = embeddings_size
        self._batch_size = 64
        self._scope = scope
        if self._args.experiment == "C":
            self._hidden_size = 1024
        else:
            self._hidden_size = 512
        self._num_proj = 512
        os.makedirs(self._args.result_dir, exist_ok=True)
        self._model_path = "{}/lstm_{}.ckpt".format(self._args.result_dir, self._args.experiment)
        self._wordToID_path = "{}/wordToID_{}.pkl".format(self._args.result_dir, self._args.experiment)
        self._max_pred_sentence_len = max_pred_sentence_len
        self._model()
    
    def _model(self):
        with tf.variable_scope(self._scope or "Model") as scope:
            # Input sentence [Batch_size, timestep]
            self._x = tf.placeholder(tf.int32, [None, self._timestep])
            # Embeddings
            self.embeddings = tf.Variable(tf.random_uniform([self._classes, self._emd_size], -0.25, 0.25))
            x = tf.nn.embedding_lookup(self.embeddings, self._x)
            # Labels
            self._y = tf.placeholder(tf.int32, [None, self._timestep])
            # LSTM
            self._cell = tf.contrib.rnn.LSTMCell(num_units=self._hidden_size,
                    initializer=tf.contrib.layers.xavier_initializer())
            outputs, _ = static_rnn(self._cell, x, dtype=tf.float32)
            # Dense layer to down-project the hidden state in experiment C
            if self._args.experiment == "C":
                outputs = tf.layers.dense(inputs=outputs, units=self._num_proj, activation=None, name="dense1")
            # Dense layer before softmax
            dense = tf.layers.dense(inputs=outputs, units=self._classes, activation=None, name="dense2")
            # Cross entropy loss and softmax
            self.y = tf.reshape(dense, [-1, self._classes])
            y_ = tf.reshape(self._y, [-1])
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=self.y)
            # Keep only non-pad elements for the loss calculation
            pad_matrix = self._wordToID["<pad>"]*tf.ones_like(y_)
            not_equals_pad = tf.cast(tf.not_equal(pad_matrix, y_), tf.float32)
            self.loss = tf.reduce_mean(softmax_loss*not_equals_pad)
            # Optimizer with clip by norm
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
            optimizer = tf.train.AdamOptimizer(learning_rate=self._args.learning_rate) # Adam Optimizer
            self._optimizer = optimizer.apply_gradients(zip(grads, tvars))
            # Evaluation of the accuracy on non-pad elements
            self.prediction = tf.to_int32(tf.argmax(self.y, 1))
            correct_prediction = tf.cast(tf.equal(self.prediction, y_), tf.float32)
            self._accuracy = tf.reduce_sum(correct_prediction*not_equals_pad) / tf.reduce_sum(not_equals_pad)

    def _create_session(self):
        # Config for Euler and GPUs
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        if self._args.euler:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
                        inter_op_parallelism_threads=self._args.threads, 
                        intra_op_parallelism_threads=self._args.threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    def train(self, train_set, validation_set):
        # Load wordToID
        wordToID_file = open(self._wordToID_path, "wb")
        pickle.dump(self._wordToID, wordToID_file)
        wordToID_file.close()
        # Initialize the saver
        saver = tf.train.Saver()
        # Load train/validation sets
        train_total_size = train_set[0].shape[0]
        val_total_size = validation_set[0].shape[0]
        train_xs = train_set[0]
        train_ys = train_set[1]
        vali_xs = validation_set[0]
        vali_ys = validation_set[1]
        train_total_batch = int(np.ceil(train_total_size / self._batch_size))
        val_total_batch = int(np.ceil(val_total_size / self._batch_size))
        # Launch the graph
        max_accuracy = 0.
        last_improve_ep = 0
        start_time = time.time()
        with self._create_session() as sess:
            f = open("{}/log_{}.txt".format(self._args.result_dir, self._args.experiment), "w")
            print("Initializing variables...")
            init = tf.global_variables_initializer()
            sess.run(init)
            if self._args.experiment in ["B", "C"]:
                embeddings_file = path_join(self._args.data_dir, 
                                            "wordembeddings-dim100.word2vec")
                load_embeddings(session=sess, 
                                wordToID=self._wordToID, 
                                embeddings=self.embeddings, 
                                embeddings_file=embeddings_file, 
                                embeddings_size=self._emd_size)
            print("Running Model")
            for epoch in range(self._args.max_epoch):
                avg_cost = 0.
                s = 0
                e = self._batch_size
                report_time = time.time()
                for i in range(train_total_batch):
                    # Load batch for training
                    batch_xs = train_xs[s:e]
                    batch_ys = train_ys[s:e]
                    # Train model on batch
                    sess.run(self._optimizer, feed_dict={self._x: batch_xs, self._y: batch_ys})
                    loss_eval = self.loss.eval(feed_dict={self._x: batch_xs, self._y: batch_ys})
                    avg_cost += loss_eval / train_total_batch
                    s = e
                    e += self._batch_size
                    if e > train_total_size:
                        break 
                    if (i + 1) % self._args.report_rate == 0:
                        report_time = time.time() - report_time
                        sps = args.report_rate*self._batch_size/report_time
                        report_message = "Epoch: {}, Trained: {:.2f}%, Loss: {:.3f}, Sentences/sec: {:.2f}".format(epoch + 1, 100*i/train_total_batch, loss_eval, sps)
                        print(report_message)
                        f.write(report_message + "\n")
                        f.flush()
                        report_time = time.time()
                    if self._args.evaluation_rate != 0 and ((i + 1) % self._args.evaluation_rate == 0):
                        s_eval = 0
                        e_eval = self._batch_size
                        accuracy = 0.
                        # Evaluation on validation set inside an epoch
                        for b in range(val_total_batch):
                            x = vali_xs[s_eval:e_eval]
                            y = vali_ys[s_eval:e_eval]
                            accuracy += self._accuracy.eval(feed_dict={self._x: x, self._y: y}) / val_total_batch
                            s_eval = e_eval
                            e_eval += self._batch_size
                            if e_eval > val_total_size:
                                break 
                        # Save the model if it achieved better accuracy on validation set
                        save_message = ""
                        if accuracy > max_accuracy:
                            model_path = saver.save(sess, self._model_path)
                            max_accuracy = accuracy
                            last_improve_ep = epoch
                            save_message = ", Model saved: {}".format(model_path)
                        evaluation_message = "Evaluation on {:.2f}% of epoch {}: Accuracy: {:.3f}{}".format(100*i/train_total_batch, epoch + 1, accuracy, save_message)
                        print(evaluation_message)
                        f.write(evaluation_message + "\n")
                        f.flush()
                s = 0
                e = self._batch_size
                accuracy = 0.
                # Evaluation on validation set after a complete epoch
                for i in range(val_total_batch):
                    x = vali_xs[s:e]
                    y = vali_ys[s:e]
                    accuracy += self._accuracy.eval(feed_dict={self._x: x, self._y: y}) / val_total_batch
                    s = e
                    e += self._batch_size
                    if e > val_total_size:
                        break
                # Save the model if it achieved better accuracy on validation set
                save_message = ""
                if accuracy > max_accuracy:
                    model_path = saver.save(sess, self._model_path)
                    max_accuracy = accuracy
                    last_improve_ep = epoch
                    save_message = ", Model saved: {}".format(model_path)
                epoch_message = "Evaluation on total epoch {}: Loss: {:.3f}, Accuracy: {:.3f}{}".format(epoch + 1, avg_cost, accuracy, save_message)
                print(epoch_message)
                f.write(epoch_message + "\n")
                f.flush()
                # Early stopping if no improvement is found for early_stopping_epochs epochs
                if epoch - last_improve_ep > self._args.early_stopping_epochs:
                    print("No improvement during the past {} epochs, stopping optimization".format(last_improve_ep))
                    break
            elapsed_time = time.time() - start_time
            elapsed_time_message = "Total training time: {:.3f} sec".format(elapsed_time)
            print(elapsed_time_message)
            f.write(elapsed_time_message + "\n")
            f.close()
    
    def test(self, test_set):
        """
            Arg:
            test_set:           test set including x and y for each sentence of length 30
            
            Calculates perplexity over the test set and writes output file in result directory
        """
        saver = tf.train.Saver()
        with self._create_session() as sess:
            output_file = "{}/group21.perplexity{}".format(self._args.result_dir, self._args.experiment)
            f = open(output_file, "w")
            # Restore trained model
            saver.restore(sess, self._model_path)
            x_ = test_set[0]
            y_ = test_set[1]
            testsize = int(x_.shape[0])
            test_total_batch = int(np.ceil(testsize/self._batch_size))
            s = 0
            e = self._batch_size
            for b in range(test_total_batch):
                x = x_[s:e]
                y = y_[s:e]
                _y_hat = self.y.eval(session=sess, feed_dict={self._x: x, self._y: y})
                y_hat = np.reshape(_y_hat, [-1, self._timestep, self._classes])
                # Compute perplexity
                for i in range(e-s):
                    p = 0.
                    count_n = 0
                    for j in range(self._timestep):
                        # The <pad> symbols (if any) are not part of the sequence
                        if y[i][j] != self._wordToID["<pad>"]: 
                            entries = y_hat[i][j]
                            exps = np.exp(entries) / np.sum(np.exp(entries), axis=0)
                            # Take the probability of the target word given history
                            p += np.log2(exps[y[i][j]]) 
                            count_n += 1
                    # Perplexity for the sentence
                    perplexity = np.power(2, -(1/max(1, count_n))*p)
                    f.write("{:.3f}\n".format(perplexity))
                    f.flush()
                # Update indices
                s = e
                if s >= testsize:
                    break
                e += self._batch_size
                if e > testsize:
                    e = testsize
        f.close()
        print("Perplexity output written on {}".format(output_file))
    
    def _incomplete_sentence_to_IDs(self, sentence):
        # We initialize it with <bos> tag
        sentence_ids = [self._wordToID["<bos>"]]  
        for word in sentence:
            if word in self._IDtoWord.values():
                sentence_ids.append(self._wordToID[word])
            else:
                # If the new word is not in the training dictionary, add <unk>
                sentence_ids.append(self._wordToID["<unk>"])  
        return sentence_ids
    
    def predict(self, incomplete_sentences_file):
        """
            Arg:
            incomplete_sentences_file:      String, filename of the incomplete sentences
                
            Generates sentences based on the current model (self._model_path) and writes output file in result directory
        """
        incomplete_sentences = open(incomplete_sentences_file, 'r').readlines()
        self._IDtoWord = {v: k for k,v in self._wordToID.items()}
        output_file = "{}/group21.continuation".format(self._args.result_dir)
        f = open(output_file, "w")
        # At each step, use the incomplete sentence or if not available,
        # use the previous predicted word.
        # Predict words using argmax(P_t-1(words))
        # incomplete_sentence: word IDs for each word in sentence
        with self._create_session() as sess:
            with tf.variable_scope(self._scope or "Model") as scope:
                # Restore the trained model
                scope.reuse_variables()
                saver = tf.train.Saver()
                saver.restore(sess, self._model_path)
                # Build the graph for predicting
                # Input placeholder
                input_x = tf.placeholder(tf.int32, [1])
                # Get embeddings for the sentence
                input_emb = tf.nn.embedding_lookup(self.embeddings, input_x)
                # State placeholder for rnn iteration
                input_state = tf.placeholder(tf.float32, [1, self._hidden_size*2])
                # RNN iteration
                with tf.variable_scope("lstm") as scope:
                    output_predictions, output_state = self._cell(input_emb, tf.split(input_state, 2, axis=1))
                # Down-project hidden state in experiment C
                if self._args.experiment == "C":
                    output_predictions = tf.layers.dense(inputs=output_predictions, units=self._num_proj, activation=None, name="dense1")
                output_predictions = tf.layers.dense(inputs=output_predictions, units=self._classes, activation=None, name="dense2")
                for sentence in incomplete_sentences:
                    cur_state = np.zeros([1, self._hidden_size*2])
                    incomplete_sentence = self._incomplete_sentence_to_IDs(sentence.split())
                    output_sentence = []
                    predicted_id = 0
                    # Run the graph over the given sentence.
                    # Predict each word using one cell, update history and input for each word.
                    for i in range(self._max_pred_sentence_len):
                        if i < len(incomplete_sentence):
                            cur_input = incomplete_sentence[i]
                        else:
                            cur_input = predicted_id
                        # One-timestep update
                        predictions, cur_state = sess.run([output_predictions, output_state],
                                                          feed_dict={input_x: [cur_input], input_state: cur_state})
                        cur_state = np.concatenate(cur_state, axis=1)
                        if i >= (len(incomplete_sentence) - 1):
                            predicted_id = np.argmax(predictions)
                            # If <unk> is predicted and we don't want it among predicted words:
                            if predicted_id == self._wordToID["<unk>"] and self._args.ignore_unk and len(predictions[0]) > 1:
                                # Get the ID of the second largest number
                                predicted_id = np.argsort(predictions)[0][-2]
                            output_sentence.append(self._IDtoWord[predicted_id])
                        else:
                            output_sentence.append(self._IDtoWord[incomplete_sentence[i + 1]])
                        if self._IDtoWord[predicted_id] == "<eos>":
                            break
                    string_sentence = " ".join(output_sentence)
                    f.write(string_sentence + "\n")
                    f.flush()
        f.close()
        print("Continuation output written on {}".format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='data directory containing the dataset and the embeddings file')
    parser.add_argument('--result_dir', type=str, default='result', 
                        help='result directory containing logs, model checkpoints and output files')
    parser.add_argument('--experiment', type=str, default='C',
                        help='A, B or C')
    parser.add_argument('--max_epoch', type=int, default=10,
                        help='maximum number of training epochs')
    parser.add_argument('--early_stopping_epochs', type=int, default=2,
                        help='number of consecutive epochs without accuracy improvement that trigger early stopping')
    parser.add_argument('--perplexity_only', action='store_true', 
                        help='compute perplexity on test set without re-training')
    parser.add_argument('--sentence_generation', action='store_true',
                        help='generates sentences based on previously trained model (task 1.2)')
    parser.add_argument('--ignore_unk', action='store_true', 
                        help='ingnore <unk> symbol predictions when generating sentences')
    parser.add_argument('--max_predicted_sentence_len', type=int, default=20,
                        help='maximum length of predicted sentence in sentence generation task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--report_rate', type=int, default=100,
                        help='number of batches trained between loss report outputs')
    parser.add_argument('--evaluation_rate', type=int, default=8000,
                        help='number of batches trained between evaluations of the accuracy on the evalutaion set, use 0 for one evaluation per epoch')
    parser.add_argument('--euler', action='store_true', 
                        help='make the experiment compatible with Euler')
    parser.add_argument('--threads', type=int, default=4,
                        help='parallelism threads, used only for experiments on Euler')
    args = parser.parse_args()
    
    print("Experiment", args.experiment)
    
    # If training is needed on experiment A, B or C, trains the model
    # Otherwise (only for preplexity calculation or the sentence generation task), restores the model from the saved files
    # For experiments A, B and C, different models are created, the restored model can be defined by the experiment argument
    
    max_pred_sentence_len = args.max_predicted_sentence_len
    if not args.perplexity_only and not args.sentence_generation:
        train_file = path_join(args.data_dir, "sentences.train")
        wordToID = construct_wordToID(train_file)
        train = load_data(train_file, wordToID)
        val_file = path_join(args.data_dir, "sentences.eval")
        val = load_data(val_file, wordToID)
    else:
        wordToID_path = "{}/wordToID_{}.pkl".format(args.result_dir, args.experiment)
        wordToID_file = open(wordToID_path, "rb")
        wordToID = pickle.load(wordToID_file)
        wordToID_file.close()
    print("Data loaded")
    trainer = Trainer(args=args,
                        wordToID=wordToID,
                        timestep=29, 
                        classes=len(wordToID), 
                        embeddings_size=100,
                        max_pred_sentence_len=max_pred_sentence_len,
                        scope="LanguageModel")
    print("Trainer established")
    if not args.perplexity_only and not args.sentence_generation:
        trainer.train(train, val)
    if args.sentence_generation:
        incomplete_sentences_file = path_join(args.data_dir, "sentences.continuation")
        print("Sentence generation using {}...".format(incomplete_sentences_file))
        trainer.predict(incomplete_sentences_file)
    else:
        test_file = path_join(args.data_dir, "sentences.test")
        test = load_data(test_file, wordToID)
        print("Computing perplexity on {}...".format(test_file))
        trainer.test(test)
