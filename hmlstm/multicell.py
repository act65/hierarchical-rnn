from .hmlstm_cell import HMLSTMCell, HMLSTMState
from .multi_hmlstm_cell import MultiHMLSTMCell

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

import numpy as np


class MultiHMLSTMCellv2(rnn_cell_impl.RNNCell):
    def __init__(self,
                 input_size=1,
                 num_outputs=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 out_hidden_size=100,
                 batch_size=50,
                 reuse=tf.AUTO_REUSE):
        """
        HMLSTMCELL is a class representing hierarchical multiscale
        long short-term memory network.

        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer correspodning to its index.
        out_hidden_size: integer, the size of the two hidden layers in the
            output network.
        batch_size: integer, the size of the batches
        """
        super(MultiHMLSTMCellv2, self).__init__(_reuse=reuse)
        self._out_hidden_size = out_hidden_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._num_outputs = num_outputs
        self._batch_size = batch_size

        if type(hidden_state_sizes) is list \
            and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the'
                             + ' same as the nubmer of layers.')

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        self.hmlstm = self.create_multicell(input_size=self._input_size, hidden_state_sizes=self._hidden_state_sizes,
            batch_size=self._batch_size,
            reuse=tf.AUTO_REUSE)

    def create_multicell(self, input_size, hidden_state_sizes, batch_size, reuse):
        def hmlstm_cell(layer):
            if layer == 0:
                h_below_size = input_size
            else:
                h_below_size = hidden_state_sizes[layer - 1]

            if layer == len(hidden_state_sizes) - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                h_above_size = hidden_state_sizes[0]
            else:
                h_above_size = hidden_state_sizes[layer + 1]

            return HMLSTMCell(hidden_state_sizes[layer], batch_size,
                              h_below_size, h_above_size, reuse)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(len(hidden_state_sizes))], reuse)

        return hmlstm

    def split_out_cell_states(self, accum):
        '''
        accum: [B, H], i.e. [B, sum(h_l) * 2 + num_layers]


        cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        '''
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]

        split_states = array_ops.split(value=accum,
                                       num_or_size_splits=splits, axis=1)

        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))

        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        '''
        hidden_states: [[B, h_l] for l in range(L)]

        h_aboves: [B, sum(ha_l)], ha denotes h_above
        '''
        concated_hs = array_ops.concat(hidden_states[1:], axis=1)

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1)

        return h_aboves

    def call(self, inputs, state):
        cell_states = self.split_out_cell_states(state)

        h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                     self._batch_size, self.hmlstm)    # [B, sum(ha_l)]
        # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
        hmlstm_in = array_ops.concat((inputs, h_aboves), axis=1)
        y, state = self.hmlstm(hmlstm_in, cell_states)
        # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]) ->
        # a list of [B, h_l + h_l + 1]
        concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
        return tf.layers.dense(tf.add_n(y), self._num_outputs), array_ops.concat(concated_states, axis=1)    # [B, H]

    def zero_state(self, batch_size, dtype=tf.float32):
        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        return tf.zeros([batch_size, elem_len]) # [B, H]

    @property
    def state_size(self):
        return (sum(self._hidden_state_sizes) * 2) + self._num_layers

    @property
    def output_size(self):
        # outputs h and z
        return self._num_outputs
