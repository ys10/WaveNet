# coding=utf-8
import argparse
import tensorflow as tf
import os
import tqdm
import numpy as np
from model import FastGenModel
from data import get_gen_dataset
from ops import load_model


def get_args():
    parser = argparse.ArgumentParser(description="Generate by WaveNet!")
    parser.add_argument("--data_path", type=str, default="./data/dataset-1024.tfrecords")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--gen_samples", type=int, default=16000)
    parser.add_argument("--gen_path", type=str, default="./fast_gen")
    return parser.parse_args()


def main():
    args = get_args()
    net = FastGenModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            # Read wave as conditions.
            # dataset = get_gen_dataset(args.data_path, args.batch_size)
            # dataset = dataset.repeat()
            # iterator = dataset.make_one_shot_iterator()
            # data = iterator.get_next()

            wave_placeholder = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.float32)
            data = {"wave": wave_placeholder}

            # Load gci labels as inputs.
            gci_labels_placeholder = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.float32)
            data["inputs"] = gci_labels_placeholder
        # build net.
        net_tensor_dic = net.build(data=data)
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # Start input enqueue threads.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init = tf.global_variables_initializer()
        sess.run(init)

        # warm-up queues
        print("init start!")
        labels_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
        wave_batch = np.ones(shape=(args.batch_size, 1), dtype=np.float32)
        labels_batch = sess.run(net_tensor_dic["init_op"], feed_dict={gci_labels_placeholder: labels_batch,
                                               wave_placeholder: wave_batch})
        # labels_batch = sess.run(net_tensor_dic["init_op"], feed_dict={gci_labels_placeholder: labels_batch})
        print("init done!")

        # # get checkpoint
        # ckpt = tf.train.get_checkpoint_state(args.save_path)
        # assert ckpt
        # saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        load_model(saver, sess, "save/")

        global_step_eval = sess.run(global_step)
        # labels_batch = np.zeros(shape=(args.batch_size, 1), dtype=np.float32)
        reuslt_batch = np.empty(shape=(args.batch_size, args.gen_samples), dtype=np.float32)
        for idx in tqdm.trange(args.gen_samples):
            labels_batch = sess.run(net_tensor_dic["detected_gci"],
                                    feed_dict={gci_labels_placeholder: labels_batch,
                                               wave_placeholder: wave_batch})
            # labels_batch = sess.run(net_tensor_dic["detected_gci"],
            #                         feed_dict={gci_labels_placeholder: labels_batch})
            reuslt_batch[:, idx] = labels_batch[:, 0]

    # save syn-ed audios
    if not os.path.exists(args.gen_path) or not os.path.isdir(args.gen_path):
        os.makedirs(args.gen_path)
    # reuslt_batch = np.int16(reuslt_batch * (1 << 15))
    for idx, result in enumerate(reuslt_batch):
        # siowav.write(os.path.join(args.gen_path, "{}_{}.wav".format(global_step_eval, idx)),
        #              data=result, rate=args.sample_rate)
        # TODO write result
        pass

    # coord.request_stop()
    # # Terminate as usual.  It is innocuous to request stop twice.
    # coord.join(threads)
    print("Congratulations!")


if __name__ == "__main__":
    main()
