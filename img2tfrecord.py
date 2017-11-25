# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import pprint
from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================#
def load_visda(input_dir, fr):
    for name in fr:
        img_name, img_label = name.split(' ')
        img = Image.open(os.path.join(input_dir, img_name))
        shape = np.array(img).shape
        yield img, shape[-1], img_label


def img2tfrecords2(output_dir, output_name, input_dir, train):
    _type = 'train.' if train else 'test.'
    output_file = os.path.join(output_dir, _type + output_name)
    print output_file
    if os.path.exists(output_file):
        os.remove(output_file)
    writer = tf.python_io.TFRecordWriter('./' + output_file)
    print 'convert...'
    # load data
    img_list = os.path.join(input_dir, 'image_list.txt')
    f = open(img_list, 'rb')
    for i, (img_raw, shape, label) in enumerate(load_visda(input_dir, f)):
        # if i == 100:
        #     break
        img_raw = img_raw.resize((256, 256))
        img_raw = np.asarray(img_raw, np.uint8)
        shape = np.array(img_raw).shape
        # if images have single channel, convert to three channel.
        if shape[-1] != 3:
            img_raw = np.reshape(img_raw, (256, 256, 1))
            img_raw = np.concatenate((img_raw, img_raw, img_raw), axis=2)
        # img_raw = np.array(img_raw).tostring()
        img_raw = img_raw.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return 0


# write images and label in tfrecord file and read them out
def img2tfrecords(output_dir, output_name, input_dir, train, label):
    _type = 'train.' if train else 'test.'
    output_file = os.path.join(output_dir, _type + output_name)
    print output_file
    if os.path.exists(output_file):
        os.remove(output_file)
    writer = tf.python_io.TFRecordWriter('./' + output_file)
    print 'convert...'
    # load data
    img_dir = os.path.join(input_dir, output_name)
    for i, img_name in enumerate(os.listdir(img_dir)):
        if i == 100:
            break
        img_path = os.path.join(FLAGS.input_dir, FLAGS.output_name + '/' + img_name)
        img_raw = Image.open(img_path)
        img_raw = img_raw.resize((256, 256))
        img_raw = np.asarray(img_raw, np.uint8)
        shape = np.array(img_raw).shape
        # when image is gray, copy three in axis.
        if shape[-1] != 3:
            img_raw = np.reshape(img_raw, (256, 256, 1))
            img_raw = np.concatenate((img_raw, img_raw, img_raw), axis=2)
        img_raw = img_raw.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return 0


def decode_from_tfrecords(output_dir, output_name, is_train, shuffle, batch_size):
    _type = 'train.' if is_train else 'test.'
    output_file = os.path.join(output_dir, _type + output_name)
    filename_queue = tf.train.string_input_producer([output_file], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    # image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int64)

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        x, y = tf.train.shuffle_batch([image, label],
                                      batch_size=batch_size,
                                      num_threads=3,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue)
    else:
        x, y = tf.train.batch([image, label],
                              batch_size=batch_size,
                              num_threads=3,
                              capacity=capacity,
                              )
    return x, y


# =============================================================================#
flags = tf.app.flags
flags.DEFINE_integer('flag', 1, "Flag to write or read [1 write, 2 read]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("target_label", 0, "The dimension of feature [5000]")
flags.DEFINE_string("output_dir", "./VISDA/", "The directory name to save data")
flags.DEFINE_string("output_name", "vis2017", "The directory name to save data")
flags.DEFINE_string("input_dir", "./train/", "The Directory name to input data")
flags.DEFINE_string("input_name", "image_list.txt", "The name to input data[books, dvd, electronics, kitchen]")
flags.DEFINE_boolean("is_shuffle", False, "True for shuffle, false for not [shuffle]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

pairs = {
    'aeroplane': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'horse': 4,
    'knife': 5,
    'motorcycle': 6,
    'person': 7,
    'plant': 8,
    'skateboard': 9,
    'train': 10,
    'truck': 11
}

flag = 1
if __name__ == '__main__':

    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    if flag == 1:
        img2tfrecords2(FLAGS.output_dir, FLAGS.output_name, FLAGS.input_dir,
                       FLAGS.is_train)
    elif flag == 2:
        img_list = os.path.join(FLAGS.input_dir, 'image_list.txt')
        with open(img_list, 'rb') as f:
            for img, shape in load_visda(FLAGS.input_dir, f):
                print np.asarray(img, np.uint8)
                plt.imshow(img)
                plt.show()
    else:
        train_data, train_label = decode_from_tfrecords(FLAGS.output_dir,
                                                        output_name=FLAGS.output_name,
                                                        is_train=FLAGS.is_train,
                                                        shuffle=FLAGS.is_shuffle,
                                                        batch_size=FLAGS.batch_size)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session(config=config) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                # while not coord.should_stop():
                for i in range(3):
                    example, l = sess.run([train_data, train_label])
                    print 'example shape: ', example[0].shape
                    exa = example[0]
                    # exa = Image.fromarray(exa, 'RGB')  # 这里Image是之前提到的
                    print 'example[0] data', exa
                    # plt.imshow(exa)
                    # plt.show()
                    print 'label[0]:', l[0]

            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()
                coord.join(threads)
