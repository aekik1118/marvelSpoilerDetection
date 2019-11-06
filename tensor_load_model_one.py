import tensorflow.compat.v1 as tf
import sys
import numpy

import text_embedding as te
import korean_splitter as ks
import korean_cnn as kc

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE

TEST_STR = "아이언11맨 사111111망"
SAVE_MODEL = 'model/model_test_4000_filter_6.ckpt'

INPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted_only_sp.csv"

textEmbedding = te.TextEmbedding(INPUT_DATA_FILE_NAME)

sp_data = ks.KoreanSplitter.split_korean(TEST_STR)

test_data = textEmbedding.embedding_string(sp_data)

cnn = kc.KoreanCNN(INPUT_ROW_SIZE, INPUT_COL_SIZE)

with tf.Session() as sess:
    print("load data...")
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_MODEL)
    print(sess.run(cnn.predictions, feed_dict={cnn.X: test_data}))
