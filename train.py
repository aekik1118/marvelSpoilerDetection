import tensorflow.compat.v1 as tf
import sys
import numpy

import text_embedding as te
import korean_cnn as kc

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE
TRAIN_DATA_SIZE = 100
TEST_DATA_SIZE = 500
batch_size = 100
training_epochs = 3

SAVE_MODEL = 'model/model_test_500_drop.ckpt'
INPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted_mini.csv"

textEmbedding = te.TextEmbedding(INPUT_DATA_FILE_NAME)

test_data, test_labels, train_data, train_labels = textEmbedding.embedding_data(TEST_DATA_SIZE, TRAIN_DATA_SIZE)

cnn = kc.KoreanCNN(INPUT_ROW_SIZE, INPUT_COL_SIZE)

train_step = tf.train.AdamOptimizer(0.005).minimize(cnn.loss)

with tf.Session() as sess:
    print("start..")
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        batch_count = int(train_data.shape[0]/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = train_data[i*batch_size:(i+1)*batch_size], train_labels[i*batch_size:(i+1)*batch_size]
            sess.run(train_step, feed_dict={cnn.X: batch_x, cnn.Y: batch_y, cnn.dropout_keep_prob: 0.5})

        print(str(epoch+1)+':'+str(sess.run(cnn.accuracy, feed_dict={cnn.X: test_data, cnn.Y: test_labels, cnn.dropout_keep_prob: 1})))

    saver = tf.train.Saver()
    saver.save(sess, SAVE_MODEL)
