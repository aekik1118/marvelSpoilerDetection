import tensorflow.compat.v1 as tf
from sklearn.preprocessing import LabelEncoder
import sys
import numpy
import random

import text_embedding as te
import korean_cnn as kc

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE
TRAIN_DATA_SIZE = 1
TEST_DATA_SIZE = 1000

SAVE_MODEL = 'model/model_30000_400_100_mini.ckpt'

INPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted_only_nosp.csv"

textEmbedding = te.TextEmbedding(INPUT_DATA_FILE_NAME)

test_data, test_labels, train_data, train_labels = textEmbedding.embedding_data(TEST_DATA_SIZE, TRAIN_DATA_SIZE)

cnn = kc.KoreanCNN(INPUT_ROW_SIZE, INPUT_COL_SIZE)

with tf.Session() as sess:
    print("load data...")
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_MODEL)
    print(sess.run(cnn.accuracy, feed_dict={cnn.X: test_data, cnn.Y: test_labels}))
