# -*- coding: utf-8 -*-
import layers
import tensorflow as tf
from network import Network


class HQNetwork(Network):

    def __init__(self, conf):
        super(HQNetwork, self).__init__(conf)

        with tf.variable_scope(self.name):
            self.num_options = 10
            self.target_ph = tf.placeholder('float32', [None], name='target')
            self.selected_option_ph = tf.placeholder('float32', [self.batch_size, self.num_options], name='selected_option')
            encoded_state = []
            diffs = []
            for i in range(self.num_options):
                with tf.variable_scope("option_{}".format(i)):
                    es = self._build_encoder()
                    encoded_state.append(es)
                    diffs.append(self._build_q_head(es))

            self.loss = self._value_function_loss(tf.reduce_sum(tf.stack(diffs, 1) * self.selected_option_ph, 1))
            self._build_gradient_ops(self.loss)


    def _build_q_head(self, input_state):
        self.w_out, self.b_out, self.output_layer = layers.fc('fc_out', input_state, self.num_actions, activation="linear")
        self.q_selected_action = tf.reduce_sum(self.output_layer * self.selected_action_ph, axis=1)

        diff = tf.subtract(self.target_ph, self.q_selected_action)
        return diff
        # return self._value_function_loss(diff)

