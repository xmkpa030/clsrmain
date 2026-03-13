# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.nn import dynamic_rnn

from reco_utils.recommender.deeprec.models.sequential.clsr import CLSRModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["CLSRDisentangleModel"]


class CLSRDisentangleModel(CLSRModel):
    """Phase-1 disentangle-ready CLSR.

    Keep original long/short modeling + alpha fusion,
    and split long/short branches for later multi-intent extension.
    """

    def _get_loss(self):
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.contrastive_loss = self._compute_contrastive_loss()
        self.discrepancy_loss = self._compute_discrepancy_loss()
        self.orth_loss = tf.multiply(self.hparams.lambda_orth, self.long_orth_loss + self.short_orth_loss)
        self.loss = self.data_loss + self.regular_loss + self.contrastive_loss + self.discrepancy_loss + self.orth_loss
        return self.loss

    def _build_seq_graph(self):
        hparams = self.hparams
        with tf.variable_scope("clsr_disentangle"):
            hist_input = tf.concat([self.item_history_embedding, self.cate_history_embedding], 2)
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            long_interest = self._build_long_interest(hist_input)
            short_interest = self._build_short_interest(hist_input)

            long_intents = self._disentangle_long_interest(long_interest)
            short_intents = self._disentangle_short_interest(short_interest)

            self.long_repr_dis = self._aggregate_long_intents(long_intents)
            self.short_repr_dis = self._aggregate_short_intents(short_intents)
            # keep original attribute names for existing loss/training compatibility
            self.att_fea_long = self.long_repr_dis
            self.att_fea_short = self.short_repr_dis

            user_embed = self._fuse_long_short(self.long_repr_dis, self.short_repr_dis)
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

    def _build_long_interest(self, hist_input):
        with tf.variable_scope("long_term"):
            att_outputs_long = self._attention_fcn(self.user_long_embedding, hist_input)
            long_interest = tf.reduce_sum(att_outputs_long, 1)
            tf.summary.histogram("att_fea_long_base", long_interest)
            self.hist_mean = tf.reduce_sum(hist_input * tf.expand_dims(self.real_mask, -1), 1) / tf.reduce_sum(
                self.real_mask, 1, keepdims=True
            )
            return long_interest

    def _build_short_interest(self, hist_input):
        hparams = self.hparams
        with tf.variable_scope("short_term"):
            if hparams.interest_evolve:
                _, short_term_intention = dynamic_rnn(
                    tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    initial_state=self.user_short_embedding,
                    dtype=tf.float32,
                    scope="short_term_intention",
                )
            else:
                short_term_intention = self.user_short_embedding
            tf.summary.histogram("GRU_final_state", short_term_intention)

            self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            self.recent_mask = tf.logical_and(self.position >= 1, self.position <= hparams.contrastive_recent_k)
            self.real_recent_mask = tf.where(
                self.recent_mask,
                tf.ones_like(self.recent_mask, dtype=tf.float32),
                tf.zeros_like(self.recent_mask, dtype=tf.float32),
            )
            self.hist_recent = tf.reduce_sum(hist_input * tf.expand_dims(self.real_recent_mask, -1), 1) / tf.reduce_sum(
                self.real_recent_mask, 1, keepdims=True
            )

            if hparams.sequential_model == "time4lstm":
                item_history_embedding_new = tf.concat(
                    [hist_input, tf.expand_dims(self.iterator.time_from_first_action, -1)], -1
                )
                item_history_embedding_new = tf.concat(
                    [item_history_embedding_new, tf.expand_dims(self.iterator.time_to_now, -1)], -1
                )
                rnn_outputs, _ = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
            elif hparams.sequential_model == "gru":
                rnn_outputs, _ = dynamic_rnn(
                    tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="simple_gru",
                )
            elif hparams.sequential_model == "lstm":
                rnn_outputs, _ = dynamic_rnn(
                    tf.nn.rnn_cell.LSTMCell(hparams.hidden_size),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="simple_lstm",
                )
            tf.summary.histogram("LSTM_outputs", rnn_outputs)

            short_term_query = tf.concat([short_term_intention, self.target_item_embedding], -1)
            att_outputs_short = self._attention_fcn(short_term_query, rnn_outputs)
            short_interest = tf.reduce_sum(att_outputs_short, 1)
            tf.summary.histogram("att_fea_short_base", short_interest)
            return short_interest

    def _disentangle_long_interest(self, long_interest):
        k_long = self.hparams.K_long
        intent_list = []
        gram_list = []
        with tf.variable_scope("long_disentangle"):
            for i in range(k_long):
                intent_delta_i = tf.layers.dense(
                    long_interest, long_interest.shape[-1].value, name="long_intent_proj_{}".format(i)
                )
                intent_i = long_interest + intent_delta_i
                intent_list.append(intent_i)
                gram_list.append(tf.nn.l2_normalize(intent_i, axis=1))
            intents = tf.stack(intent_list, axis=1)
            if k_long > 1:
                gram = tf.matmul(tf.stack(gram_list, axis=1), tf.stack(gram_list, axis=1), transpose_b=True)
                eye = tf.eye(k_long, batch_shape=[tf.shape(long_interest)[0]])
                self.long_orth_loss = tf.reduce_mean(tf.square(gram - eye))
            else:
                self.long_orth_loss = tf.constant(0.0)
            return intents

    def _disentangle_short_interest(self, short_interest):
        k_short = self.hparams.K_short
        intent_list = []
        gram_list = []
        with tf.variable_scope("short_disentangle"):
            for i in range(k_short):
                intent_delta_i = tf.layers.dense(
                    short_interest, short_interest.shape[-1].value, name="short_intent_proj_{}".format(i)
                )
                intent_i = short_interest + intent_delta_i
                intent_list.append(intent_i)
                gram_list.append(tf.nn.l2_normalize(intent_i, axis=1))
            intents = tf.stack(intent_list, axis=1)
            if k_short > 1:
                gram = tf.matmul(tf.stack(gram_list, axis=1), tf.stack(gram_list, axis=1), transpose_b=True)
                eye = tf.eye(k_short, batch_shape=[tf.shape(short_interest)[0]])
                self.short_orth_loss = tf.reduce_mean(tf.square(gram - eye))
            else:
                self.short_orth_loss = tf.constant(0.0)
            return intents

    def _aggregate_long_intents(self, long_intents):
        with tf.variable_scope("long_intent_aggregate"):
            long_gate_logits = tf.layers.dense(long_intents, 1, name="long_gate_score")
            long_gate_logits = tf.squeeze(long_gate_logits, axis=-1)
            long_gate_weights = tf.nn.softmax(long_gate_logits, axis=1)
            long_interest = tf.reduce_sum(long_intents * tf.expand_dims(long_gate_weights, -1), axis=1)

            long_gate_entropy = -tf.reduce_mean(
                tf.reduce_sum(long_gate_weights * tf.log(long_gate_weights + 1e-12), axis=1)
            )

            tf.summary.histogram("long_gate_logits", long_gate_logits)
            tf.summary.histogram("long_gate_weights", long_gate_weights)
            tf.summary.scalar("long_gate_entropy", long_gate_entropy)
        tf.summary.histogram("att_fea_long", long_interest)
        return long_interest

    def _aggregate_short_intents(self, short_intents):
        with tf.variable_scope("short_intent_aggregate"):
            short_gate_logits = tf.layers.dense(short_intents, 1, name="short_gate_score")
            short_gate_logits = tf.squeeze(short_gate_logits, axis=-1)
            short_gate_weights = tf.nn.softmax(short_gate_logits, axis=1)
            short_interest = tf.reduce_sum(short_intents * tf.expand_dims(short_gate_weights, -1), axis=1)

            short_gate_entropy = -tf.reduce_mean(
                tf.reduce_sum(short_gate_weights * tf.log(short_gate_weights + 1e-12), axis=1)
            )

            tf.summary.histogram("short_gate_logits", short_gate_logits)
            tf.summary.histogram("short_gate_weights", short_gate_weights)
            tf.summary.scalar("short_gate_entropy", short_gate_entropy)
        tf.summary.histogram("att_fea_short", short_interest)
        return short_interest

    def _fuse_long_short(self, long_repr, short_repr):
        hparams = self.hparams
        with tf.name_scope("alpha"):
            hist_input = tf.concat([self.item_history_embedding, self.cate_history_embedding], 2)
            if not hparams.manual_alpha:
                if hparams.predict_long_short:
                    with tf.variable_scope("causal2"):
                        _, final_state = dynamic_rnn(
                            tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
                            inputs=hist_input,
                            sequence_length=self.sequence_length,
                            dtype=tf.float32,
                            scope="causal2",
                        )
                        tf.summary.histogram("causal2", final_state)

                    concat_all = tf.concat(
                        [
                            final_state,
                            self.target_item_embedding,
                            long_repr,
                            short_repr,
                            tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                        ],
                        1,
                    )
                else:
                    concat_all = tf.concat(
                        [
                            self.target_item_embedding,
                            long_repr,
                            short_repr,
                            tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                        ],
                        1,
                    )

                alpha_logit = self._fcn_net(concat_all, hparams.att_fcn_layer_sizes, scope="fcn_alpha")
                self.alpha_output = tf.sigmoid(alpha_logit)
                user_embed = long_repr * self.alpha_output + short_repr * (1.0 - self.alpha_output)
                tf.summary.histogram("alpha", self.alpha_output)
                self.alpha_output_mean = self.alpha_output
                error_with_category = self.alpha_output_mean - self.iterator.attn_labels
                tf.summary.histogram("error_with_category", error_with_category)
                squared_error_with_category = tf.math.sqrt(
                    tf.math.squared_difference(
                        tf.reshape(self.alpha_output_mean, [-1]), tf.reshape(self.iterator.attn_labels, [-1])
                    )
                )
                tf.summary.histogram("squared_error_with_category", squared_error_with_category)
            else:
                self.alpha_output = tf.constant([[hparams.manual_alpha_value]])
                user_embed = long_repr * hparams.manual_alpha_value + short_repr * (1.0 - hparams.manual_alpha_value)
            return user_embed

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("contrastive_loss", self.contrastive_loss)
        tf.summary.scalar("discrepancy_loss", self.discrepancy_loss)
        tf.summary.scalar("orth_loss", self.orth_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged
