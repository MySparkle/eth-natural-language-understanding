from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def static_rnn(cell, inputs, dtype, scope=None):
    """
        Args:
            cell:       LSTM cell
            inputs:     numpy array of size (batch_size, 30, embedding_size), input IDs for the model
            dtype:      type of the hidden state of the LSTM
            scope:      tensorflow variable scope for LSTM variables

        Returns:
            outputs:    numpy array of size (batch_size, 30, hidden_size), outputs of each timestep
            state:      final state of the LSTM
    """
    with tf.variable_scope(scope or "lstm") as scope:
        batch_size = tf.shape(inputs)[0]
        timestep = int(inputs.shape[1])
        state = cell.zero_state(batch_size, dtype=dtype)
        inputs = tf.unstack(inputs, axis=1)
        outputs = []
        for i in range(timestep):
            if i > 0:
                scope.reuse_variables()
            output, state = cell(inputs[i], state)
            outputs.append(output)
        outputs = tf.stack(outputs, axis=1)
    return outputs, state
