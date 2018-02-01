# coding=utf-8
from builtins import object

import tensorflow as tf

from layers import dilated_causal_conv1d, fast_gen_dilated_causal_conv1d, self_gated_layer
from audio import mu_law
from loss import focal_loss


class Model(object):
    def __init__(self, name="WaveNet"):
        self.name = name
        self.max_global_steps = 200000
        self.blks = 3
        self.layers_per_blk = 10
        self.conv_width = 2
        self.dilation_base = 2
        self.residual_units = 256
        self.skip_units = 256
        self.hidden_out = 256
        self.sample_classes = 256
        self.label_classes = 2
        self.central_class = self.sample_classes // 2
        self.label_pad_value = 0

    @property
    def reception_fields(self):
        return (self.conv_width - 1) * (self.dilation_base ** self.layers_per_blk - 1) * self.blks + 1

    def build(self, data, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            wave = data["wave"]
            mu_wav = mu_law(wave)
            conditions = tf.cast(mu_wav + self.central_class, dtype=tf.int32)
            inputs = tf.to_int64(data["labels"])

            # 1st. build the initial causal layer.
            with tf.variable_scope("init_causal_layer"):
                # 1st. right shift inputs to get labels.
                labels = tf.pad(inputs[:, :-1], [[0, 0], [1, 0]], constant_values=self.label_pad_value)
                # 2nd. to one-hot representation
                conditions_one_hot = tf.one_hot(conditions, depth=self.sample_classes, dtype=tf.float32)
                inputs_one_hot = tf.one_hot(inputs, depth=self.label_classes, dtype=tf.float32)
                # labels_one_hot = tf.one_hot(labels, depth=self.label_classes, dtype=tf.float32)
                # 3rd. calculate
                init_causal_out = dilated_causal_conv1d(inputs=inputs_one_hot, dilation_rate=1,
                                                        filters=self.residual_units, width=self.conv_width)

            # 2nd. build the dilated causal blocks.
            with tf.variable_scope("dilated_causal_blocks"):
                resi_out = init_causal_out
                skip_out = 0.
                for idx in range(self.blks):
                    with tf.variable_scope("blk_{}".format(idx)):
                        for idy in range(self.layers_per_blk):
                            with tf.variable_scope("layer_{}".format(idy)):
                                conv_out = dilated_causal_conv1d(inputs=resi_out,
                                                                 dilation_rate=self.dilation_base ** idy,
                                                                 filters=2*self.residual_units,
                                                                 width=self.conv_width)
                                conditions_conv = tf.layers.dense(inputs=conditions_one_hot,
                                                                  units=2*self.residual_units,
                                                                  activation=None, name="conditions_conv")
                                acti_out = self_gated_layer(inputs=tf.add(conv_out, conditions_conv))
                                resi_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="residual_out")
                                skip_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="skip_out")

            # 3rd. calculate probabilities, and get loss.
            with tf.variable_scope("softmax"):
                # 1st. relu
                skip_out = tf.nn.relu(skip_out)
                # 2nd. dense -> relu
                skip_out = tf.layers.dense(inputs=skip_out, units=self.hidden_out, activation=tf.nn.relu)
                # 3rd. dense -> softmax
                logits = tf.layers.dense(inputs=skip_out, units=self.label_classes, activation=tf.nn.softmax)
                # 4th. get modes
                modes = tf.argmax(logits, axis=-1)
                # 5th. get loss.
                loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            return {"wave": wave, "output": tf.to_float(modes), "labels": tf.to_float(labels), "loss": loss}


class FocalLossModel(object):
    def __init__(self, name="WaveNet"):
        self.name = name
        self.max_global_steps = 200000
        self.blks = 3
        self.layers_per_blk = 10
        self.conv_width = 2
        self.dilation_base = 2
        self.residual_units = 256
        self.skip_units = 256
        self.hidden_out = 256
        self.sample_classes = 256
        self.label_classes = 2
        self.central_class = self.sample_classes // 2
        self.label_pad_value = 0

    @property
    def reception_fields(self):
        return (self.conv_width - 1) * (self.dilation_base ** self.layers_per_blk - 1) * self.blks + 1

    def build(self, data, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            wave = data["wave"]
            mu_wav = mu_law(wave)
            conditions = tf.cast(mu_wav + self.central_class, dtype=tf.int32)
            inputs = tf.to_int64(data["labels"])

            # 1st. build the initial causal layer.
            with tf.variable_scope("init_causal_layer"):
                # 1st. right shift inputs to get labels.
                labels = tf.pad(inputs[:, :-1], [[0, 0], [1, 0]], constant_values=self.label_pad_value)
                # 2nd. to one-hot representation
                conditions_one_hot = tf.one_hot(conditions, depth=self.sample_classes, dtype=tf.float32)
                inputs_one_hot = tf.one_hot(inputs, depth=self.label_classes, dtype=tf.float32)
                labels_one_hot = tf.one_hot(labels, depth=self.label_classes, dtype=tf.float32)
                # 3rd. calculate
                init_causal_out = dilated_causal_conv1d(inputs=inputs_one_hot, dilation_rate=1,
                                                        filters=self.residual_units, width=self.conv_width)

            # 2nd. build the dilated causal blocks.
            with tf.variable_scope("dilated_causal_blocks"):
                resi_out = init_causal_out
                skip_out = 0.
                for idx in range(self.blks):
                    with tf.variable_scope("blk_{}".format(idx)):
                        for idy in range(self.layers_per_blk):
                            with tf.variable_scope("layer_{}".format(idy)):
                                conv_out = dilated_causal_conv1d(inputs=resi_out,
                                                                 dilation_rate=self.dilation_base ** idy,
                                                                 filters=2*self.residual_units,
                                                                 width=self.conv_width)
                                conditions_conv = tf.layers.dense(inputs=conditions_one_hot,
                                                                  units=2*self.residual_units,
                                                                  activation=None, name="conditions_conv")
                                acti_out = self_gated_layer(inputs=tf.add(conv_out, conditions_conv))
                                resi_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="residual_out")
                                skip_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="skip_out")

            # 3rd. calculate probabilities, and get loss.
            with tf.variable_scope("softmax"):
                # 1st. relu
                skip_out = tf.nn.relu(skip_out)
                # 2nd. dense -> relu
                skip_out = tf.layers.dense(inputs=skip_out, units=self.hidden_out, activation=tf.nn.relu)
                # 3rd. dense -> softmax
                logits = tf.layers.dense(inputs=skip_out, units=self.label_classes, activation=tf.nn.softmax)
                # 4th. get modes
                modes = tf.argmax(logits, axis=-1)
                # 5th. get focal loss.
                loss = focal_loss(labels=labels_one_hot, logits=logits)

            return {"wave": wave, "output": tf.to_float(modes), "labels": tf.to_float(labels), "loss": loss}


class FastGenModel(object):
    def __init__(self, name="WaveNet"):
        self.name = name
        self.max_global_steps = 200000
        self.blks = 3
        self.layers_per_blk = 10
        self.conv_width = 2
        self.dilation_base = 2
        self.residual_units = 256
        self.skip_units = 256
        self.hidden_out = 256
        self.sample_classes = 256
        self.label_classes = 2
        self.central_class = self.sample_classes // 2
        self.label_pad_value = 0

    @property
    def reception_fields(self):
        return (self.conv_width - 1) * ((self.dilation_base ** self.layers_per_blk - 1) * self.blks + 1) + 1

    def build(self, data, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            wave = data["wave"]
            # assert wave.get_shape().with_rank(2)
            # assert wave.get_shape()[1].value == 1
            mu_wav = mu_law(wave)
            conditions = tf.cast(mu_wav + self.central_class, dtype=tf.int32)
            # TODO construct inputs by the shape of conditions.
            inputs = tf.to_int64(data["inputs"])
            assert inputs.get_shape().with_rank(2)
            assert inputs.get_shape()[1].value == 1

            init_op_lst = []
            # 1st. build the initial causal layer.
            with tf.variable_scope("init_causal_layer"):
                # 1st. to one-hot representation
                conditions_one_hot = tf.one_hot(conditions, depth=self.sample_classes, dtype=tf.float32)
                inputs_one_hot = tf.one_hot(inputs, depth=self.label_classes, dtype=tf.float32)
                # 3rd. calculate
                init_causal_out = fast_gen_dilated_causal_conv1d(inputs=inputs_one_hot, dilation_rate=1,
                                                                 filters=self.residual_units, width=self.conv_width)
                init_op_lst.append(init_causal_out["init_op"])
                init_causal_out = init_causal_out["out"]

            # 2nd. build the dilated causal blocks.
            with tf.variable_scope("dilated_causal_blocks"):
                resi_out = init_causal_out
                skip_out = 0.
                for idx in range(self.blks):
                    with tf.variable_scope("blk_{}".format(idx)):
                        for idy in range(self.layers_per_blk):
                            with tf.variable_scope("layer_{}".format(idy)):
                                conv_out = fast_gen_dilated_causal_conv1d(inputs=resi_out,
                                                                          dilation_rate=self.dilation_base ** idy,
                                                                          filters=2*self.residual_units,
                                                                          width=self.conv_width)
                                conditions_conv = tf.layers.dense(inputs=conditions_one_hot,
                                                                  units=2*self.residual_units,
                                                                  activation=None, name="conditions_conv")
                                init_op_lst.append(conv_out["init_op"])
                                conv_out = conv_out["out"]
                                acti_out = self_gated_layer(inputs=tf.add(conv_out, conditions_conv))
                                resi_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="residual_out")
                                skip_out += tf.layers.dense(inputs=acti_out, units=self.residual_units,
                                                            activation=None, name="skip_out")

            # 3rd. calculate probabilities, and get loss.
            with tf.variable_scope("softmax"):
                # 1st. relu
                skip_out = tf.nn.relu(skip_out)
                # 2nd. dense -> relu
                skip_out = tf.layers.dense(inputs=skip_out, units=self.hidden_out, activation=tf.nn.relu)

                # 3rd. dense -> softmax
                logits = tf.layers.dense(inputs=skip_out, units=self.label_classes, activation=tf.nn.softmax)
                # 4th. get modes
                modes = tf.argmax(logits, axis=-1)
                # # 3rd. dense, get energy(logits).
                # logits = tf.layers.dense(inputs=skip_out, units=self.label_classes, activation=None)
                # # 4th. sample from the logits, then recover to waveform := [-1, 1).
                # logits = tf.squeeze(logits, axis=1)
                # synthesized_samples = tf.multinomial(logits=logits, num_samples=1) - self.central_class
                # synthesized_samples = inv_mu_law(synthesized_samples)

            return {"init_op": init_op_lst, "detected_gci": modes}
