import tensorflow.compat.v1 as tf
import sys
import numpy

class KoreanCNN(object):
    def __init__(self, INPUT_ROW_SIZE, INPUT_COL_SIZE,):
        tf.disable_v2_behavior()
        numpy.set_printoptions(threshold=sys.maxsize)

        self.X = tf.placeholder(tf.float32, [None, INPUT_ROW_SIZE, INPUT_COL_SIZE, 1])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        self.filter_sizes = [3, 4, 5, 6, 7, 8]

        embedding_size = INPUT_COL_SIZE
        num_filters = 128
        sequence_length = INPUT_ROW_SIZE
        num_classes = 2
        l2_reg_lambda = 0.0
        l2_loss = tf.constant(0.0)
        # dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.X,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"
            )
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # with tf.name_scope("dropout"):
        #     h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes]
                # initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Combine all the pooled features
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.Y)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        self.train_step = tf.train.AdamOptimizer(0.005).minimize(self.loss)
