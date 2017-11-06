# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from utils import load_amazon


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


def read():
    # Read TFRecords file
    # current_path = os.getcwd()
    save_data_dir = 'data/Amazon_review/'
    tfrecords_file_name = "train.books"
    input_file = os.path.join(save_data_dir, tfrecords_file_name)

    # Constrain the data to print
    max_print_number = 2002
    print_number = 1

    for serialized_example in tf.python_io.tf_record_iterator(input_file):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # Read data in specified format
        label = example.features.feature["Y"].float_list.value
        features = example.features.feature["X"].float_list.value
        # print("Number: {}, label: {}, features: {}".format(print_number, label,
        #                                                    features))
        # print np.array(features).shape
        if np.array(features).shape == (5000,):
            pass
        else:
            print 'wrong'
            exit()
        # Return when reaching max print number
        if print_number > max_print_number:
            exit()
        else:
            print_number += 1


def decode_from_tfrecords(output_dir, output_name, shuffle, batch_size):
    _type = 'train.' if train else 'test.'
    output_file = os.path.join(output_dir, _type + output_name)
    filename_queue = tf.train.string_input_producer([output_file], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 'X': tf.FixedLenFeature([5000], tf.float32),
                                           # 'Y': tf.FixedLenFeature([], tf.float32)
                                           'X': tf.FixedLenFeature([5000], tf.int64),
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
flag = 3
train = True
if __name__ == '__main__':
    output_dir = 'data/Amazon_review/'
    input_dir = './data/'

    input_name = 'books'
    input_suffix = '_train.svmlight' if train else '_test.svmlight'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xs, ys = load_amazon(input_name, input_dir, input_suffix)

    if flag == 1:
        text2tfrecords(output_dir, input_name, train, xs, ys)
    elif flag == 2:
        read()
    elif flag == 3:

        train_data, train_label = decode_from_tfrecords(output_dir,
                                                        output_name=input_name,
                                                        shuffle=False,
                                                        batch_size=64)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                # while not coord.should_stop():
                for i in range(3):
                    example, l = sess.run([train_data, train_label])
                    print 'example: ', example[0].shape
                    print example[0]
                    print l[0]
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()
                coord.join(threads)
    else:
        pass
