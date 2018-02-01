import argparse
import tensorflow as tf
import os
import tqdm
from model import Model
from data import get_train_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Train WaveNet!")
    parser.add_argument("--data_path", type=str, default="./data/dataset-1024.tfrecords")
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length", type=int, default=2048)
    parser.add_argument("--sample_rate", type=int, default=20000)
    parser.add_argument("--add_audio_summary_per_steps", type=int, default=1000)
    parser.add_argument("--save_per_steps", type=int, default=5000)
    return parser.parse_args()


def main():
    args = get_args()
    net = Model()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            dataset = get_train_dataset(args.data_path, args.batch_size, args.crop_length)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            data = iterator.get_next()
        # build net.
        net_tensor_dic = net.build(data=data)

        # get summaries.
        audio_summary = tf.summary.merge([tf.summary.audio("wave", net_tensor_dic["wave"], args.sample_rate),
                                          tf.summary.audio("output", net_tensor_dic["output"], args.sample_rate),
                                          tf.summary.audio("labels", net_tensor_dic["labels"], args.sample_rate)
                                          ])
        loss_summary = tf.summary.scalar("loss", net_tensor_dic["loss"])

        # get optimizer.
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        opt = tf.train.AdamOptimizer(1e-4)
        upd = opt.minimize(net_tensor_dic["loss"], global_step=global_step)

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # get checkpoint
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        assert ckpt
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        else:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        summary_writer = tf.summary.FileWriter(args.log_path)
        save_path = os.path.join(args.save_path, net.name)

        global_step_eval = sess.run(global_step)
        pbar = tqdm.tqdm(total=args.steps)
        pbar.update(global_step_eval)
        while global_step_eval < args.steps:
            loss_summary_eval, audio_summary_eval, global_step_eval, _ = sess.run([loss_summary,
                                                                                   audio_summary, global_step, upd])
            summary_writer.add_summary(loss_summary_eval, global_step=global_step_eval)
            if global_step_eval % args.add_audio_summary_per_steps == 0:
                summary_writer.add_summary(audio_summary_eval, global_step=global_step_eval)
            if global_step_eval % args.save_per_steps == 0:
                if not os.path.exists(args.save_path) or not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                saver.save(sess=sess, save_path=save_path, global_step=global_step_eval)
            pbar.update(1)
        summary_writer.close()

    print("Congratulations!")


if __name__ == "__main__":
    main()
