import numpy
import tensorflow as tf
from glob import glob
import os
from utils import save_images
import time
from utils import get_image
import matplotlib.pyplot as plt


def _bytes_feature(value):
    # attention please, the parameter has the '[]'
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    # attention please, the parameter has the '[]'
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to_tfrecords(images_file, output_file, crop_dim):
    data = glob(os.path.join(images_file, '*.jpg'))
    writer = tf.python_io.TFRecordWriter(output_file)
    for index in range(len(data)):
        image_raw = get_image(data[index], crop_dim).tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "features": _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(tfrecord, w_dim, c_dim):
    # print tfrecord
    # attention please, the generate queue function' parameter need []
    filename_queue = tf.train.string_input_producer([tfrecord])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'features': tf.FixedLenFeature([], tf.string)
        }
    )
    images = tf.decode_raw(features['features'], tf.uint8)
    images = tf.reshape(images, [w_dim, w_dim, c_dim])
    return images

def get_tfrecord_batch_data(images, batch_size=64):
    # img_batch = tf.train.shuffle_batch([images], batch_size=batch_size, capacity=300000, min_after_dequeue=1000)
    img_batch = tf.train.batch([images], batch_size=batch_size, capacity=300000)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        # both parameter can't be reduced.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        samples = []
        for index in range(1):
            image = sess.run(img_batch)
            # samples.append(image)
            # print image.shape
            # break
            save_images(image, [8, 8],
                        '{}/train_{:04f}.png'.format("samples", time.time()))
    coord.request_stop()
    coord.join(threads)

def plot_data(images):
    img_batch = tf.train.shuffle_batch([images], batch_size=2, capacity=20000, min_after_dequeue=1000)
    img_batch = tf.train.batch([images], batch_size=64)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        batch = tf.train.batch([images], batch_size=3, capacity=20000)

        coord = tf.train.Coordinator()
        # both parameter can't be reduced.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sample_images = sess.run(batch)
        print len(sample_images)

        for i in range(3):
            train_images = sess.run(batch)
            plt.imshow(train_images[0])
            plt.show()
            print len(train_images)
        train_batch = tf.train.batch([images], batch_size=3, capacity=20000)
        train_images = sess.run(train_batch)
        print len(train_images)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    start = time.time()
    data_path = "/storage/wenhua/data/img_align_celeba/"
    # data_path = "/storage/user_data/wenhua/data/faces_test/"
    data_file = "faces_new.tfrecords"

    flag = 3
    if flag == 1:
        convert_to_tfrecords(data_path, data_file, crop_dim=108)
    elif flag == 2:
        images = read_and_decode(data_file, w_dim=64, c_dim=3)
        plot_data(images)
    elif flag == 3:
        images = read_and_decode(data_file, w_dim=64, c_dim=3)
        get_tfrecord_batch_data(images)
    end = time.time()
    print "elapsed :", (end - start)