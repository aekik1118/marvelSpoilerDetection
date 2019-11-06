from sklearn.preprocessing import LabelEncoder
import sys
import numpy
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TextEmbedding(object):

    letters = "0123456789ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ¡!@#%^&*()?~:;<>×_+=- /.·,´`|\"\'\\[]{}\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    INPUT_ROW_SIZE = 300
    INPUT_COL_SIZE = len(letters)

    def __init__(self, data_file_name):
        self.data_file_name = data_file_name
        self.csv_data_ohe = []

    def embedding_string(self, splitted_korean_str_list):
        INPUT_ROW_SIZE = TextEmbedding.INPUT_ROW_SIZE
        INPUT_COL_SIZE = TextEmbedding.INPUT_COL_SIZE

        letters_list = (list)(TextEmbedding.letters)
        le = LabelEncoder().fit(letters_list)

        data_str = splitted_korean_str_list

        str_size = len(data_str)
        if str_size > INPUT_ROW_SIZE:
            data_str = data_str[:INPUT_ROW_SIZE]
        elif str_size < INPUT_ROW_SIZE:
            data_str += " " * (INPUT_ROW_SIZE - str_size)

        str_list = list(data_str)

        le_letter = le.transform(str_list)
        le_letter = le_letter.reshape(-1, 1)
        ohe_letters = tf.keras.utils.to_categorical(le_letter, num_classes=INPUT_COL_SIZE)

        return numpy.array(ohe_letters).reshape(1, INPUT_ROW_SIZE, INPUT_COL_SIZE, 1)


    def embedding_data(self, TEST_DATA_SIZE, TRAIN_DATA_SIZE):

        INPUT_ROW_SIZE = TextEmbedding.INPUT_ROW_SIZE
        INPUT_COL_SIZE = TextEmbedding.INPUT_COL_SIZE

        letters_list = (list)(TextEmbedding.letters)
        le = LabelEncoder().fit(letters_list)

        with open(self.data_file_name, "r", encoding='UTF8') as file:
            csv_data_input = []
            ohe_list = []
            for line in file.readlines():
                csv_data_input.append(line.split(','))

            csv_data = random.sample(csv_data_input, len(csv_data_input))

            tlarr = []

            for i in csv_data:
                if i[0][0] == '0':
                    tlarr.append([1, 0])
                else:
                    tlarr.append([0, 1])

            train_labels = numpy.array(tlarr)

            for i in csv_data:
                data_str = i[1]

                str_size = len(data_str)
                if str_size > INPUT_ROW_SIZE:
                    data_str = data_str[:INPUT_ROW_SIZE]
                elif str_size < INPUT_ROW_SIZE:
                    data_str += " " * (INPUT_ROW_SIZE - str_size)

                str_list = list(data_str)

                le_letter = le.transform(str_list)
                le_letter = le_letter.reshape(-1, 1)
                ohe_letters = tf.keras.utils.to_categorical(le_letter, num_classes=INPUT_COL_SIZE)

                ohe_list.append(ohe_letters)

            train_data = numpy.array(ohe_list).reshape(len(csv_data_input), INPUT_ROW_SIZE, INPUT_COL_SIZE, 1)

            test_data = train_data[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE + TEST_DATA_SIZE]
            test_labels = train_labels[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE + TEST_DATA_SIZE]

            train_data = train_data[:TRAIN_DATA_SIZE]
            train_labels = train_labels[:TRAIN_DATA_SIZE]

            print(test_data.shape)
            print(test_labels.shape)

            print((train_data).shape)
            print((train_labels).shape)

            return test_data, test_labels, train_data, train_labels



