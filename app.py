from flask import Flask, render_template, request, jsonify
import tensorflow.compat.v1 as tf
import sys
import numpy

import text_embedding as te
import korean_splitter as ks
import korean_cnn as kc

app = Flask(__name__)

INPUT_ROW_SIZE = te.TextEmbedding.INPUT_ROW_SIZE
INPUT_COL_SIZE = te.TextEmbedding.INPUT_COL_SIZE

SAVE_MODEL = 'model/model_test_4000_filter_6.ckpt'
textEmbedding = te.TextEmbedding("")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/spoiler', methods = ['POST'])
def spoiler():
    data = request.get_json()
    print(data['content'])
    raw_data = data['content']

    sp_data = ks.KoreanSplitter.split_korean(raw_data)
    test_data = textEmbedding.embedding_string(sp_data)

    cnn = kc.KoreanCNN(INPUT_ROW_SIZE, INPUT_COL_SIZE)

    with tf.Session() as sess:
        print("load data...")
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_MODEL)
        res = sess.run(cnn.predictions, feed_dict={cnn.X: test_data})
        print(res)

        if res[0] == 1:
            return jsonify({'tof': 'true'})

    return jsonify({'tof':'false'})
