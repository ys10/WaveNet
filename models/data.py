import tensorflow as tf


def parse_single_example(record):
    features = tf.parse_single_example(record,
                                       features={
                                           "wave": tf.FixedLenFeature([], tf.string),
                                           "labels": tf.FixedLenFeature([], tf.string),
                                       })
    wave = tf.to_float(tf.decode_raw(features["wave"], tf.int64))
    labels = tf.decode_raw(features["labels"], tf.float32)
    return {"wave": wave, "labels": labels}


def crop_wav(crop_length):
    def __crop(inputs):
        wave = tf.random_crop(inputs["wave"], size=[crop_length])
        wave.set_shape([crop_length])
        labels = tf.random_crop(inputs["labels"], size=[crop_length])
        labels.set_shape([crop_length])
        return {"wave": wave, "labels": labels}
    return __crop


def get_dataset(tfrecord_path, batch_size=16, crop_length=16000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.map(crop_wav(crop_length))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    return dataset
