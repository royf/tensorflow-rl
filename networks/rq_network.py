# -*- coding: utf-8 -*-
import layers
import tensorflow as tf
from network import Network


class RQNetwork(Network):

    def __init__(self, conf):
        super(RQNetwork, self).__init__(conf)
                
        with tf.variable_scope(self.name):
            self.num_rews = 10
            self.target_ph = tf.placeholder('float32', [None], name='target')
            self.rew_cnt_ph = tf.placeholder('int32', [self.batch_size], name='rew_cnt')
            encoded_state = self._build_encoder(tf.one_hot(self.rew_cnt_ph, self.num_rews))

            self.loss = self._build_q_head(encoded_state)
            self._build_gradient_ops(self.loss)


    def _build_q_head(self, input_state):
        self.w_out, self.b_out, self.output_layer = layers.fc('fc_out', input_state, self.num_actions, activation="linear")
        self.q_selected_action = tf.reduce_sum(self.output_layer * self.selected_action_ph, axis=1)

        diff = tf.subtract(self.target_ph, self.q_selected_action)
        return self._value_function_loss(diff)

