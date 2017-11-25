# Notes for learning tensorflow

### 1. convert_to_tfrecords.py

convert raw data to tfrecords, which is a binary file for tensorflow. It can take good use of the memory, cache etc, and fastly copy, move, read from tensorflow.

#### here are some steps:

**make tfrecords data:**

1. loading data and convert string type, and use Example object encode data

**decode data (read data)**

1. make filename queue
2. parse and decode data into another queue
3. return batch data(shuffle or not shuffle)


## 2. text2records.py

There are many methods tell us how to convert images to tfrecords, but only a few method says how to convert text to records. So, here show how to do it:

data: Amazon review

### step

2.1 genenrate tfrecords:

python text2records.py —flag=1

2.2 read data:

python text2records.py —flag=2

**note that:** 

Here is the Amazon review data, if you want to use your own text data, you only need to modify here:

(1). load_amazon() function in the main() : load your own data

(2). dim_feature parameter in flags: the your own feature dimension

## 3. load_pretrain_model

code in load_pretrain_model directory

when training some complex network(VGG, resnet), we often need to load the pretrain model to init network in order to train fastly. Here is the example for resnet101:

Int the my_model.py is our network structure, and run the main.py to load model by python main.py.














