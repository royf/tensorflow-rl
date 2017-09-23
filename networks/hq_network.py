# -*- coding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from network import Network


class HQNetwork(Network):

    def __init__(self, conf):
        super(HQNetwork, self).__init__(conf)

        with tf.variable_scope(self.name):
            self.num_options = 10
            self.target_ph = tf.placeholder('float32', [None], name='target')
            self.selected_option_ph = tf.placeholder('int32', [self.batch_size], name='selected_option')
            vd = VarDispenser(676915, self.num_options, self.selected_option_ph)
            encoded_state = self._build_encoder(vd)

            self.loss = self._build_q_head(vd, encoded_state)
            self._build_gradient_ops(self.loss)
            assert vd.exhausted()


    def _build_q_head(self, vd, input_state):
        self.w_out, self.b_out, self.output_layer = layers.fc('fc_out', vd, input_state, self.num_actions, activation="linear")
        self.q_selected_action = tf.reduce_sum(self.output_layer * self.selected_action_ph, axis=1)

        diff = tf.subtract(self.target_ph, self.q_selected_action)
        return self._value_function_loss(diff)


class VarDispenser(object):
    def __init__(self, num_vars, num_options, option_selector):
        self.num_vars = num_vars
        self.all_vars = tf.get_variable('all_vars', [num_options, self.num_vars], tf.float32, self.initializer)
        self.option_vars = tf.matmul(tf.one_hot(option_selector, num_options), self.all_vars)
        self.next_index = 0
        self.inits = []

    def get_variable(self, name, shape, dtype, initializer):
        nvars = np.prod(shape)
        v = tf.reshape(self.option_vars[:, self.next_index:self.next_index+nvars], [-1] + shape)
        print(name, v.shape, nvars)
        self.next_index += nvars
        self.inits.append((initializer, shape))
        return v

    def initializer(self, shape, dtype, partition_info):
        def init():
            return tf.get_default_session().run(tf.concat([tf.reshape(init(shape, dtype, partition_info), [-1]) for init, shape in self.inits], 0))
        return tf.reshape(tf.py_func(init, [], tf.float32, False), shape)

    def exhausted(self):
        print(self.next_index, self.num_vars)
        return self.next_index == self.num_vars
