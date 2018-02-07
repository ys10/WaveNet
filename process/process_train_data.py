# coding=utf-8
import tensorflow as tf
import numpy as np

from ops import file_names, read_wave_data, read_marks_data, make_mask, mask_wave, mask_marks_1d

marks_path = "data/marks/"
wave_path = "data/wave/"
marks_extension = ".marks"
wave_extension = ".wav"
data_path = "data/training.tfrecords"
crop_length = 8000
mask_range = crop_length


def training_data_feature(wave, labels):
    feature_dict = {
        "wave": tf.train.Feature(bytes_list=tf.train.BytesList(value=[wave.tobytes()])),
        "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
    }
    return feature_dict


def main():
    with tf.python_io.TFRecordWriter(data_path) as writer:
        print("Process training data start!")
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
            mask = make_mask(marks_data, wave_length, mask_range=mask_range)
            # mask wave & marks data
            masked_wave = mask_wave(wave_data, mask)
            masked_marks = mask_marks_1d(marks_data, mask, wave_length)
            assert len(masked_marks) == len(masked_wave)
            print("key: {}, wave_length: {}, number of marks: {}, number of sub sequences: {}"
                  .format(key, wave_length, len(marks_data), len(masked_wave)))
            # write to TFRecords file.
            for i in range(len(masked_marks)):
                wave = masked_wave[i]
                labels = masked_marks[i]
                if len(wave) < crop_length:
                    print("Warning: length of wave is less than crop length, key: {}, index: {}, length: {}"
                          .format(key, i, len(wave)))
                example = tf.train.Example(features=tf.train.Features(feature=training_data_feature(wave, labels)))
                writer.write(example.SerializeToString())
        print("Done!")


if __name__ == '__main__':
    main()
