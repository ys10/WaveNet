# coding=utf-8

import argparse
import tensorflow as tf
import os
import tqdm

from model import FocalLossModel, Model
from dis_model import FocalLossDisModel, DisModel
from data import get_training_dataset
from model_loader import load_model, save_model


def get_args():
    parser = argparse.ArgumentParser(description="WaveNet!")
    parser.add_argument("--training_data_path", type=str, default="./data/training.tfrecords")
    parser.add_argument("--validation_data_path", type=str, default="./data/validation.tfrecords")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--max_to_keep", type=int, default=500)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--training_steps", type=int, default=250000)
    parser.add_argument("--validation_size", type=int, default=152)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_length", type=int, default=8000)
    parser.add_argument("--sample_rate", type=int, default=20000)
    parser.add_argument("--add_audio_summary_per_steps", type=int, default=1000)
    parser.add_argument("--save_per_steps", type=int, default=5000)
    parser.add_argument("--validation_per_steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = get_args()
    net = DisModel()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("training_data"):
            training_set = get_training_dataset(args.training_data_path, args.batch_size, args.crop_length)
            training_set = training_set.repeat()
            training_iterator = training_set.make_one_shot_iterator()
            training_data = training_iterator.get_next()

        with tf.variable_scope("validation_data"):
            validation_set = get_training_dataset(args.validation_data_path, args.validation_size, args.crop_length)
            validation_set = validation_set.repeat()
            validation_iterator = validation_set.make_one_shot_iterator()
            validation_data = validation_iterator.get_next()
        # build net.
        training_tensor_dic = net.build(data=training_data, reuse=tf.AUTO_REUSE)
        validation_tensor_dic = net.build(data=validation_data, reuse=tf.AUTO_REUSE)

        # get summaries.
        audio_summary = tf.summary.merge([tf.summary.audio("wave", training_tensor_dic["wave"], args.sample_rate),
                                          tf.summary.audio("output", training_tensor_dic["output"], args.sample_rate),
                                          tf.summary.audio("labels", training_tensor_dic["labels"], args.sample_rate)
                                          ])
        training_loss_summary = tf.summary.scalar("training_loss", training_tensor_dic["loss"])
        validation_loss_summary = tf.summary.scalar("validation_loss", validation_tensor_dic["loss"])

        # get optimizer.
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        opt = tf.train.AdamOptimizer(1e-3)
        upd = opt.minimize(training_tensor_dic["loss"], global_step=global_step)

        # get saver.
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(graph=graph, config=config) as sess:
        save_path = os.path.join(args.save_path, net.name)
        if not load_model(saver, sess, save_path):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        training_writer = tf.summary.FileWriter(args.log_path + "/training", sess.graph)
        validation_writer = tf.summary.FileWriter(args.log_path + "/validation")

        global_step_eval = sess.run(global_step)
        pbar = tqdm.tqdm(total=args.training_steps)
        pbar.update(global_step_eval)
        while global_step_eval < args.training_steps:
            training_loss_summary_eval, audio_summary_eval, global_step_eval, _ = \
                sess.run([training_loss_summary, audio_summary, global_step, upd])
            training_writer.add_summary(training_loss_summary_eval, global_step=global_step_eval)
            """summary audio"""
            # if global_step_eval % args.add_audio_summary_per_steps == 0:
            #     summary_writer.add_summary(audio_summary_eval, global_step=global_step_eval)
            """validate"""
            if global_step_eval % args.validation_per_steps == 0:
                validation_loss_summary_eval = sess.run(validation_loss_summary)
                tf.logging.info("Validation done.")
                validation_writer.add_summary(validation_loss_summary_eval, global_step=global_step_eval)
            """save model"""
            if global_step_eval % args.save_per_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_model(saver, sess, save_path, global_step_eval)
            pbar.update(1)
        training_writer.close()
        validation_writer.close()

    print("Congratulations!")


if __name__ == "__main__":
    main()
