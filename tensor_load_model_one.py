import tensorflow.compat.v1 as tf
from sklearn.preprocessing import LabelEncoder
import sys
import numpy
import random

import text_embedding as te
import korean_splitter as ks

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE
TRAIN_DATA_SIZE = 1
TEST_DATA_SIZE = 5

TEST_STR = "아이언맨 멋지다"

SAVE_MODEL = 'model/model_test.ckpt'

tf.disable_v2_behavior()

INPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted_only_sp.csv"

numpy.set_printoptions(threshold=sys.maxsize)

textEmbedding = te.TextEmbedding(INPUT_DATA_FILE_NAME)

sp_data = ks.KoreanSplitter.split_korean(TEST_STR)

test_data = textEmbedding.embedding_string(sp_data)

# test_data, test_labels, train_data, train_labels = textEmbedding.embedding_data(TEST_DATA_SIZE, TRAIN_DATA_SIZE)

X = tf.placeholder(tf.float32, [None, INPUT_ROW_SIZE, INPUT_COL_SIZE,1])
Y = tf.placeholder(tf.float32, [None, 2])
filter_sizes = [3, 4, 5]
embedding_size = INPUT_COL_SIZE
num_filters = 128
sequence_length = INPUT_ROW_SIZE
num_classes = 2
l2_reg_lambda = 0.0
l2_loss = tf.constant(0.0)
# dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1,shape=[num_filters]), name="b")
    conv = tf.nn.conv2d(
        X,
        W,
        strides=[1,1,1,1],
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
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
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
    scores = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
# Combine all the pooled features
losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y)
loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
# correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
# Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predictions))
train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    print("load data...")
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_MODEL)
    print(sess.run(predictions, feed_dict={X: test_data}))
