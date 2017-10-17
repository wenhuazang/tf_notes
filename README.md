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















