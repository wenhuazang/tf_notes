import tensorflow as tf
import my_model
import tensorflow.contrib.slim as slim
import os
import numpy as np

checkpoints_dir = 'pretrain_model'
checkpoints_name = 'resnet_v1_101.ckpt'
pretrained_model = 'resnet_v1_101'


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["resnet_v1_101/logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_101.ckpt'),
        variables_to_restore)


# Create the model, use the default arg scope to configure the batch norm parameters.
processed_images = tf.constant(np.ones([32, 224, 224, 3]), dtype=tf.float32)
one_hot_labels = tf.one_hot(np.ones([32, ]), 12)

with slim.arg_scope(my_model.resnet_arg_scope()):
    logits, _ = my_model.resnet_v1_101(processed_images, num_classes=12, is_training=True)
pred = tf.nn.softmax(logits)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if pretrained_model:
        print('[!] Restoring pretrained model: %s' % pretrained_model)
        init_fn = get_init_fn()
        init_fn(sess)
    # your code #######
    p = sess.run(pred)
    print p
    ###################
