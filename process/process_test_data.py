# coding=utf-8
import tensorflow as tf
import numpy as np

from ops import file_names, read_wave_data, read_marks_data, make_full_mask, mask_wave, mask_marks_1d

marks_path = "data/marks/"
wave_path = "data/wave/"
marks_extension = ".marks"
wave_extension = ".wav"
data_path = "data/testing.tfrecords"


def testing_data_feature(wave, length, key, rate):
    feature_dict = {
        "wave": tf.train.Feature(bytes_list=tf.train.BytesList(value=[wave.tobytes()])),
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
        "key": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(key, encoding="utf-8")])),
        "rate": tf.train.Feature(int64_list=tf.train.Int64List(value=[rate])),
    }
    return feature_dict


def main():
    with tf.python_io.TFRecordWriter(data_path) as writer:
        print("Process test data start!")
        keys = file_names(marks_path)
        print("file number: {}".format(len(keys)))
        for key in keys:
            # wave data.
            rate, wave_data = read_wave_data(wave_path + key + wave_extension)
            wave_data = wave_data.astype(np.int64)
            wave_length = len(wave_data)
            # marks data.
            marks_data = read_marks_data(marks_path + key + marks_extension, rate, wave_length)
            # mask
            mask = make_full_mask(wave_length)
            # mask full wave & marks data
            masked_wave = mask_wave(wave_data, mask)
            masked_marks = mask_marks_1d(marks_data, mask, wave_length)
            assert len(masked_marks) == len(masked_wave)
            # print data info
            print("key: {}, wave_length: {}, number of marks: {}, number of sub sequences: {}"
                  .format(key, wave_length, len(marks_data), len(masked_wave)))
            # write to TFRecords file.
            wave = masked_wave[0]
            labels = masked_marks[0]
            example = tf.train.Example(features=tf.train.Features(
                feature=testing_data_feature(wave, wave_length, key, rate)))
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
