import glob
import json
import random

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from tqdm import tqdm
import numpy as np
import os
from main import read_coco

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image_path, model):
    image = preprocess(image_path)
    fea_vec = model.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


def greedySearch(photo, max_length, wordtoindex, indextoword, model):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoindex[w] for w in in_text.split() if w in wordtoindex]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = indextoword[str(yhat)]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def beam_search_predictions(image, max_length, wordtoindex, indextoword, model, beam_index=3):
    start = [wordtoindex["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [indextoword[str(i)] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

if __name__ == '__main__':
    model_inceptionv3 = InceptionV3(weights='imagenet')
    model_inceptionv3 = Model(model_inceptionv3.input, model_inceptionv3.layers[-2].output)

    with open('temp_file/20000/train_img.json', 'r') as f:
        json_data = json.load(f)
    wordtoindex = json_data['wordtoindex']
    indextoword = json_data['indextoword']
    max_length = json_data['max_length']

    if os.path.exists('temp.json'):
        with open('temp.json', 'r') as f:
            test_img_info = json.load(f)
    else:
        test_img_info = read_coco('MSCOCO/annotations/captions_val2017.json')
        with open('temp.json', 'w') as f:
            json.dump(test_img_info, f)


    test_img = [img_path.replace('\\', '/') for img_path in glob.glob('MSCOCO/val2017/*.jpg')]
    encoding_test = {}
    pbar = tqdm(test_img)
    for img_path in pbar:
        pbar.set_description('Test encode')
        img_name = img_path.split('/')[-1]
        if os.path.exists(img_path + '.npy'):
            encoding_test[img_name] = np.load(img_path + '.npy')
        else:
            encoding_test[img_name] = encode(img_path, model_inceptionv3)
            np.save(img_path + '.npy', encoding_test[img_name])

    # 选择使用的神经网络
    model = load_model('models/model_20000_300_20220408_1115.h5')

    # 单张图片进行测试
    # image_path = 'MSCOCO/val2017/000000001584.jpg'
    # image = encode(image_path, model_inceptionv3).reshape((1, 2048))
    # print(encoding_test[pic])
    # print(image)
    # print(np.load('MSCOCO/val2017/{}.npy'.format(pic)).reshape((1, 2048)))
    # # image = encoding_test[pic].reshape((1, 2048))
    # image = np.array(np.random.random((1, 2048))).reshape((1, 2048))
    # x = plt.imread('MSCOCO/val2017/{}'.format(pic))
    # plt.imshow(x)
    # plt.show()
    #
    # print("Greedy Search:", greedySearch(image, max_length, wordtoindex, indextoword, model))
    # print("Beam Search, K = 3:", beam_search_predictions(image, wordtoindex, indextoword, model, beam_index=3))
    # print("Beam Search, K = 5:", beam_search_predictions(image, wordtoindex, indextoword, model, beam_index=5))
    # print("Beam Search, K = 7:", beam_search_predictions(image, wordtoindex, indextoword, model, beam_index=7))
    # print("Beam Search, K = 10:", beam_search_predictions(image, wordtoindex, indextoword, model, beam_index=10))

    # 多张图片进行测试
    # npic = 8
    # npix = 299
    # target_size = (npix, npix, 3)
    # count = 1
    #
    # fig = plt.figure(figsize=(10, 20))
    # for img_path in tqdm(random.sample(test_img, 8)):
    #     image = np.load(img_path+'.npy').reshape((1, 2048))
    #     original_captions = list(test_img_info[img_path.split('/')[-1]])[:3]
    #     BS_3 = beam_search_predictions(image, wordtoindex, indextoword, model, beam_index=3)
    #
    #     captions = [img_path.split('/')[-1], BS_3]
    #     captions.extend(original_captions)
    #     image = load_img(img_path, target_size=target_size)
    #     ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
    #     ax.imshow(image)
    #     count+=1
    #
    #     ax = fig.add_subplot(npic, 2, count)
    #     plt.axis('off')
    #     ax.plot()
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 5)
    #     for i, caption in enumerate(captions):
    #         ax.text(0, i, caption, fontsize=20)
    #     count+=1
    # plt.show()

    # 为所有测试图片生成语句
    all_img_info = []
    name_to_id = {}
    with open('MSCOCO/annotations/captions_val2017.json', 'r') as f:
        json_data = json.load(f)
    for image in json_data['images']:
        name_to_id[image['file_name']] = image['id']

    for img_path in tqdm(test_img):
        img_name = img_path.split('/')[-1]
        temp_dic = {}
        temp_dic['image_id'] = name_to_id[img_name]
        temp_dic['caption'] = beam_search_predictions(encoding_test[img_name].reshape((1, 2048)), wordtoindex,
                                           indextoword, model)
        all_img_info.append(temp_dic)
    with open('MSCOCO/annotations/captions_val2017_fakecap_results.json', 'w') as f:
        json.dump(all_img_info, f)

