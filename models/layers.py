import tensorflow as tf


def self_gated_layer(inputs, name=None):
    with tf.name_scope(name, default_name="self_gated_layer", values=[inputs]):
        # TODO self gated layer based on even channels.
        assert inputs.get_shape()[-1].value % 2 == 0
        energy_l, energy_r = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.nn.sigmoid(energy_l) * tf.nn.tanh(energy_r)


def dilated_causal_conv1d(inputs, width, dilation_rate, filters, name="dilated_causal_conv1d", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert inputs.get_shape().with_rank(3)
        pad_len = (width - 1) * dilation_rate
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        return tf.layers.conv1d(inputs=padded_inputs,
                                dilation_rate=dilation_rate,
                                kernel_size=width,
                                strides=1,
                                filters=filters,
                                padding="valid")


def dilated_causal_conv1d_no_pad(inputs, width, dilation_rate, filters, name="dilated_causal_conv1d", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert inputs.get_shape().with_rank(3)
        return tf.layers.conv1d(inputs=inputs,
                                dilation_rate=dilation_rate,
                                kernel_size=width,
                                strides=1,
                                filters=filters,
                                padding="valid")


def fast_gen_dilated_causal_conv1d(inputs, width, dilation_rate, filters, name="dilated_causal_conv1d", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert inputs.get_shape().with_rank(3)
        assert inputs.get_shape()[1].value == 1
        assert width >= 2

        batch_size = tf.shape(inputs)[0]
        channels_in = inputs.get_shape()[-1].value
        # create queue.
        queue = tf.FIFOQueue(capacity=dilation_rate, dtypes=[tf.float32] * (width - 1))

        # init op. must be run at the very first.
        init_op = queue.enqueue_many([[tf.zeros(shape=(batch_size, 1, channels_in),
                                                dtype=tf.float32)] * dilation_rate] * (width - 1))

        # get pre-trained weights.
        kernel = tf.get_variable(name="conv1d/kernel", shape=(width, channels_in, filters), dtype=tf.float32)
        bias = tf.get_variable(name="conv1d/bias", shape=filters, dtype=tf.float32)

        # get real inputs.
        deque_out = queue.dequeue()
        if width == 2:
            out = tf.matmul(deque_out[:, 0, :], kernel[0]) + tf.matmul(inputs[:, 0, :], kernel[1]) + bias
            out = tf.reshape(out, shape=(batch_size, 1, filters), name=None)
            deque_out = [deque_out]
        elif width == 3:
            out = tf.matmul(deque_out[0][:, 0, :], kernel[0]) + tf.matmul(deque_out[1][:, 0, :], kernel[1]) + \
                  tf.matmul(inputs[:, 0, :], kernel[2]) + bias
        else:
            raise NotImplementedError

        # update op.
        upd_op = queue.enqueue(deque_out[1:] + [inputs])

        with tf.control_dependencies([upd_op]):
            controlled_out = tf.identity(out)

        return {"init_op": init_op, "out": controlled_out}
