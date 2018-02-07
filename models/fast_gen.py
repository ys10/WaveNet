# coding=utf-8

import argparse
import tensorflow as tf
import os
import tqdm
import numpy as np

from model import FastGenModel
from data import get_testing_dataset
from model_loader import load_model
from process.ops import trans_labels2marks, save_marks


def get_args():
    parser = argparse.ArgumentParser(description="Generate by WaveNet!")
    parser.add_argument("--data_path", type=str, default="./data/testing.tfrecords")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gen_samples", type=int, default=16000)
    parser.add_argument("--gen_path", type=str, default="./fast_gen")
    parser.add_argument("--file_num", type=int, default=10)
    return parser.parse_args()


def read_from_data_set(data_path, batch_size):
    data_set = get_testing_dataset(data_path, batch_size)
    data_set = data_set.repeat()
    iterator = data_set.make_one_shot_iterator()
    return iterator.get_next()


def main():
    args = get_args()
    net = FastGenModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            next_example = read_from_data_set(args.data_path, args.batch_size)
            wave_placeholder = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.float32)
            gci_labels_placeholder = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.float32)
            data = {"wave": wave_placeholder, "inputs": gci_labels_placeholder}
        # build net.
        net_tensor_dic = net.build(data=data)

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(graph=graph, config=config) as sess:
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # init = tf.global_variables_initializer()
        # sess.run(init)

        for i in range(args.file_num):
            init = tf.global_variables_initializer()
            sess.run(init)

            value = sess.run(next_example)
            length = value["length"][0]
            wave = value["wave"]
            key = value["key"][0].decode('utf-8')
            rate = value["rate"][0]
            tf.logging.info("key: {}, wave_length: {}".format(key, length))

            labels_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
            sess.run(net_tensor_dic["init_op"],
                     feed_dict={gci_labels_placeholder: labels_batch,
                                wave_placeholder: wave[:, 0:1]})
            # get checkpoint
            save_path = os.path.join(args.save_path, net.name)
            load_model(saver, sess, save_path)
            result_batch = np.empty(shape=(args.batch_size, length), dtype=np.float32)
            for time_step in tqdm.trange(length):
                labels_batch = sess.run(net_tensor_dic["detected_gci"],
                                        feed_dict={gci_labels_placeholder: labels_batch,
                                                   wave_placeholder: wave[:, time_step:time_step + 1]})
                result_batch[:, time_step] = labels_batch[:, 0]
            # Make generate directory if it doesn't exist.
            if not os.path.exists(args.gen_path) or not os.path.isdir(args.gen_path):
                os.makedirs(args.gen_path)
            for idx, result in enumerate(result_batch):
                marks = trans_labels2marks(result, rate=rate)
                marks_path = os.path.join(args.gen_path, key+".marks")
                tf.logging.info("Save marks to " + marks_path)
                save_marks(marks_path, marks)
                pass
            pass

    #     # warm-up queues
    #     labels_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
    #     wave_batch = np.ones(shape=(args.batch_size, 1), dtype=np.float32)
    #     tf.logging.info("init start!")
    #     sess.run(net_tensor_dic["init_op"],
    #              feed_dict={gci_labels_placeholder: labels_batch, wave_placeholder: wave_batch})
    #     tf.logging.info("init done!")
    #
    #     # get checkpoint
    #     save_path = os.path.join(args.save_path, net.name)
    #     load_model(saver, sess, save_path)
    #
    #     global_step_eval = sess.run(global_step)
    #     # labels_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
    #     reuslt_batch = np.empty(shape=(args.batch_size, args.gen_samples), dtype=np.float32)
    #     for idx in tqdm.trange(args.gen_samples):
    #         labels_batch = sess.run(net_tensor_dic["detected_gci"],
    #                                 feed_dict={gci_labels_placeholder: labels_batch,
    #                                            wave_placeholder: wave_batch})
    #         # labels_batch = sess.run(net_tensor_dic["detected_gci"],
    #         #                         feed_dict={gci_labels_placeholder: labels_batch})
    #         reuslt_batch[:, idx] = labels_batch[:, 0]
    #
    # # save syn-ed audios
    # if not os.path.exists(args.gen_path) or not os.path.isdir(args.gen_path):
    #     os.makedirs(args.gen_path)
    # # reuslt_batch = np.int16(reuslt_batch * (1 << 15))
    # for idx, result in enumerate(reuslt_batch):
    #     # siowav.write(os.path.join(args.gen_path, "{}_{}.wav".format(global_step_eval, idx)),
    #     #              data=result, rate=args.sample_rate)
    #     # TODO write result
    #     pass
    coord.request_stop()
    # Terminate as usual.  It is innocuous to request stop twice.
    coord.join(threads)
    print("Congratulations!")


if __name__ == "__main__":
    main()
