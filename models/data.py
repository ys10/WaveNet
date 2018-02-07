import tensorflow as tf


def parse_training_example(record):
    features = tf.parse_single_example(record,
                                       features={
                                           "wave": tf.FixedLenFeature([], tf.string),
                                           "labels": tf.FixedLenFeature([], tf.string),
                                       })
    wave = tf.to_float(tf.decode_raw(features["wave"], tf.int64))
    labels = tf.decode_raw(features["labels"], tf.float32)
    return {"wave": wave, "labels": labels}


def parse_testing_example(record):
    features = tf.parse_single_example(record,
                                       features={
                                           "wave": tf.FixedLenFeature([], tf.string),
                                           "length": tf.FixedLenFeature([], tf.int64),
                                           "key": tf.FixedLenFeature([], tf.string),
                                           "rate": tf.FixedLenFeature([], tf.int64),
                                       })
    wave = tf.to_float(tf.decode_raw(features["wave"], tf.int64))
    length = tf.cast(features["length"], tf.int32)
    key = features["key"]
    rate = tf.cast(features["rate"], tf.int32)
    return {"wave": wave, "length": length, "key": key, "rate": rate}


def crop_training_data(crop_length):
    def __crop(inputs):
        wave = tf.random_crop(inputs["wave"], size=[crop_length])
        wave.set_shape([crop_length])
        labels = tf.random_crop(inputs["labels"], size=[crop_length])
        labels.set_shape([crop_length])
        return {"wave": wave, "labels": labels}
    return __crop


def get_training_dataset(tfrecord_path, batch_size=16, crop_length=16000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_training_example)
    dataset = dataset.map(crop_training_data(crop_length))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    return dataset


def get_testing_dataset(tfrecord_path, batch_size=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_testing_example)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    return dataset
