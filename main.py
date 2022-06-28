import random

import keras.models
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import pickle
import string
import os
import cv2
import glob
from PIL import Image
from time import time

from keras import Input, layers
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout, BatchNormalization
from keras.layers import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

import json
from utils import *
import datetime
from show_json_tree import show_json


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image, model):
    image = preprocess(image)
    fea_vec = model.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def read_coco(annotation_file, num=None):
    '''
    从json文件中读取图片名称和标题，并返回以图片名称为索引的字典
    :param annotation_file: coco annotation file path
    :param num: number of images for training, set 'None' for all images
    :return: all_img_infos
    '''
    with open(annotation_file, 'r') as f:
        json_data = json.load(f)

    all_img_name = {}
    all_image_info = {}
    images = json_data['images']
    if num==None:
        random.shuffle(images)
    else:
        images = random.sample(images, num)
    pbar = tqdm(images)
    for image in pbar:
        pbar.set_description('Read image from {} file'.format(annotation_file))
        all_img_name[image['id']] = image['file_name']

    pbar = tqdm(json_data['annotations'])
    for anno in pbar:
        pbar.set_description('Read anno from {} file'.format(annotation_file))
        image_id = anno['image_id']
        if image_id not in all_img_name:
            continue
        caption = anno['caption']
        image_name = all_img_name[image_id]
        if image_name not in all_image_info:
            all_image_info[image_name] = list()
        all_image_info[image_name].append(caption)

    return all_image_info

if __name__ == '__main__':
    coco_dir = 'MSCOCO'
    data_type = 'captions'
    train_dir = 'train'
    val_dir = 'val'
    train_annotation_file = '{}/annotations/{}_{}2017.json'.format(coco_dir, data_type, train_dir)

    num = 20000 #选择用来作为训练集的图片的数量，None表示全部

    #如果本地存在已保存的文件，则直接读取
    if os.path.exists('json/train_img_{}.json'.format(num)):
        with open('json/train_img_{}.json'.format(num), 'r') as f:
            json_data = json.load(f)
        train_img_info = json_data['train_img_info']
        vocab_size = json_data['vocab_size']
        indextoword = json_data['indextoword']
        for key in indextoword.keys():  indextoword[int(key)] = indextoword.pop(key)
        wordtoindex = json_data['wordtoindex']
        max_length = json_data['max_length']
        train_img = json_data['train_img']
        embedding_dim = json_data['embedding_dim']
        embedding_matrix = np.load('json/embedding_matrix_{}.npy'.format(num))
    #否则从源文件中读取处理，并保存到本地
    else:
        json_data = {}

        train_img_info = read_coco(train_annotation_file, num)

        # 文本处理，转换成小写，去除标点符号，单字符和数字值
        train_img_info = text_clean(train_img_info)

        # 为所有标题头尾添加startseq和endseq
        train_img_info = add_start_end(train_img_info)

        json_data['train_img_info'] = train_img_info

        # 创建词库，将出现频次小于10次的词从词库中删除
        vocab = make_vocab(train_img_info, threshold=10)
        vocab_size = len(vocab) + 1

        json_data['vocab_size'] = vocab_size

        # 为所有单词建立映射
        indextoword = {}
        wordtoindex = {}
        index = 1
        for w in vocab:
            wordtoindex[w] = index
            indextoword[index] = w
            index += 1
        json_data['indextoword'] = indextoword
        json_data['wordtoindex'] = wordtoindex

        # 找出标题的最大单词数
        max_length = 0
        for key in train_img_info:
            for caption in list(train_img_info[key]):
                max_length = max(max_length, len(list(caption.split())))
        json_data['max_length'] = max_length

        # 读取训练图片路径和测试图片路径
        train_img = [img_path.replace('\\', '/') for img_path in
                     glob.glob('{}/{}2017/*.jpg'.format(coco_dir, train_dir))]
        json_data['train_img'] = train_img

        # Glove嵌入，将单词转换为200维的向量
        embeddings_index = {}
        f = open('{}/glove.6B.200d.txt'.format(coco_dir), encoding="utf-8")
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_dim = 200
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in tqdm(wordtoindex.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        json_data['embedding_dim'] = embedding_dim
        np.save('json/embedding_matrix_{}.npy'.format(num), embedding_matrix)

        with open('json/train_img_{}.json'.format(num), 'w') as f:
            json.dump(json_data, f)

    # 模型建立与训练
    model_incptV3 = InceptionV3(weights='imagenet')
    model_incptV3 = Model(model_incptV3.input, model_incptV3.layers[-2].output)


    encoding_train = {}
    pbar = tqdm(train_img)
    for img_path in pbar:
        pbar.set_description('Train encode')
        img_name = img_path.split('/')[-1]
        if img_name in train_img_info.keys():
            if os.path.exists(img_path + '.npy'):
                encoding_train[img_name] = np.load(img_path + '.npy')
            else:
                encoding_train[img_name] = encode(img_path, model_incptV3)
                np.save(img_path + '.npy', encoding_train[img_name])
        else:
            continue

    train_features = encoding_train

    inputs1 = Input(shape=(2048,))
    # fe1 = Dropout(0.5)(inputs1)
    fe1 = Dense(512, activation='relu')(inputs1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    # se2 = Dropout(0.5)(se1)
    se2 = LSTM(512)(se1)

    decoder1 = add([fe1, se2])
    decoder2 = Dense(512, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.summary()

    model.get_layer('embedding').set_weights([embedding_matrix])
    model.get_layer('embedding').trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    def generator(train_img_info, photos, wordtoindex, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key, caption_list in train_img_info.items():
                n += 1
                photo = photos[key]
                for caption in caption_list:
                    seq = [wordtoindex[word] for word in caption.split(' ') if word in wordtoindex]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)

                if n == num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0


    epochs = 300
    batch_size = 8
    steps = len(train_img_info) // batch_size
    generator = generator(train_img_info, train_features, wordtoindex, max_length, batch_size)

    timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if not os.path.exists('models/{}'.format(timestr)):
        os.makedirs('models/{}'.format(timestr))

    filepath = 'models/{}/'.format(timestr) + 'model_{epoch:02d}.h5'
    check_pointer = ModelCheckpoint(filepath, verbose=1, save_weights_only=False, period=1)

    history = model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
                        callbacks=[check_pointer])



    model.save('models/model_{}_{}_{}.h5'.format(num, epochs, timestr))

    with open('json/history_{}.txt'.format(timestr), 'wb') as f:
        pickle.dump(history.history, f)


