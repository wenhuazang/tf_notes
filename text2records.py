# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from utils import load_amazon
import pprint


# =============================================================================#
def text2tfrecords(output_dir, output_name, train, X, Y):
    _type = 'train.' if train else 'test.'
    output_file = os.path.join(output_dir, _type+output_name)
    if os.path.exists(output_file):
        os.remove(output_file)
    writer = tf.python_io.TFRecordWriter('./' + output_file)

    for j, (x, y) in enumerate(zip(X, Y)):
        x = [int(i) for i in x]
        example = tf.train.Example(features=tf.train.Features(
            feature={
                # 'X': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                # 'Y': tf.train.Feature(float_list=tf.train.FloatList(value=[y]))
                'X': tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                'Y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
            }))
        writer.write(example.SerializeToString())
    writer.close()


def decode_from_tfrecords(output_dir, output_name, dim, is_train, shuffle, batch_size):
    _type = 'train.' if is_train else 'test.'
    output_file = os.path.join(output_dir, _type + output_name)
    filename_queue = tf.train.string_input_producer([output_file], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 'X': tf.FixedLenFeature([5000], tf.float32),
                                           # 'Y': tf.FixedLenFeature([], tf.float32)
                                           'X': tf.FixedLenFeature([dim], tf.int64),
                                           'Y': tf.FixedLenFeature([], tf.int64)
                                       })
    x = features['X']
    y = features['Y']
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        x, y = tf.train.shuffle_batch([x, y],
                                      batch_size=batch_size,
                                      num_threads=3,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue)
    else:
        x, y = tf.train.batch([x, y],
                              batch_size=batch_size,
                              num_threads=3,
                              capacity=capacity,
                              )
    return x, y

# =============================================================================#
flags = tf.app.flags
flags.DEFINE_integer('flag', 1, "Flag to write or read [1 write, 2 read]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("dim_feature", 5000, "The dimension of feature [5000]")
flags.DEFINE_string("output_dir", "data/Amazon_review/", "The directory name to save data")
flags.DEFINE_string("input_dir", "./data/", "The Directory name to input data")
flags.DEFINE_string("input_name", "books", "The name to input data")
flags.DEFINE_boolean("is_shuffle", True, "True for shuffle, false for not [shuffle]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    input_suffix = '_train.svmlight' if FLAGS.is_train else '_test.svmlight'

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    if FLAGS.flag == 1:
        # load data
        xs, ys = load_amazon(FLAGS.input_name, FLAGS.input_dir, input_suffix)
        text2tfrecords(FLAGS.output_dir, FLAGS.input_name, FLAGS.is_train, xs, ys)
    elif FLAGS.flag == 2:
        train_data, train_label = decode_from_tfrecords(FLAGS.output_dir,
                                                        output_name=FLAGS.input_name,
                                                        dim=FLAGS.dim_feature,
                                                        is_train=FLAGS.is_train,
                                                        shuffle=FLAGS.is_shuffle,
                                                        batch_size=FLAGS.batch_size)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                # while not coord.should_stop():
                for i in range(1):
                    example, l = sess.run([train_data, train_label])
                    print 'example shape: ', example[0].shape
                    print 'example[0] data', example[0]
                    print 'label[0]:', l[0]
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()
                coord.join(threads)
    else:
        print 'flag should be 1 or 2 !'

if __name__ == '__main__':
    tf.app.run()