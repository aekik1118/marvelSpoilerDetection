import tensorflow.compat.v1 as tf
import sys
import numpy

import text_embedding as te
import korean_cnn as kc

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE
TRAIN_DATA_SIZE = 100
TEST_DATA_SIZE = 500

SAVE_MODEL = 'model/model_test_500_test.ckpt'
INPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted.csv"

textEmbedding = te.TextEmbedding(INPUT_DATA_FILE_NAME)

test_data, test_labels, train_data, train_labels = textEmbedding.embedding_data(TEST_DATA_SIZE, TRAIN_DATA_SIZE)

cnn = kc.KoreanCNN(INPUT_ROW_SIZE, INPUT_COL_SIZE)

train_step = tf.train.AdamOptimizer(0.005).minimize(cnn.loss)

with tf.Session() as sess:
    print("start..")
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_step, feed_dict={cnn.X: train_data, cnn.Y:train_labels})
        if i%10 == 0:
            print(sess.run(cnn.accuracy, feed_dict={cnn.X: test_data, cnn.Y: test_labels}))
    saver = tf.train.Saver()
    saver.save(sess, SAVE_MODEL)
