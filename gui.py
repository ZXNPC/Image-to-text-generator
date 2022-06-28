import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy as np
from PIL import ImageTk, Image
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.preprocessing import sequence

import json


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
    # 选择神经网络
    model = load_model('models/model_20000_300_20220408_1115.h5')

    model_InceptionV3 = InceptionV3(weights='imagenet')
    model_InceptionV3 = Model(model_InceptionV3.input, model_InceptionV3.layers[-2].output)
    top = tk.Tk()
    top.geometry('800x600')
    top.title('图文生成')
    label = Label(top)
    original_image = Label(top)

    # 读取json数据
    with open('json/train_img_20000.json', 'r') as f:
        json_data = json.load(f)
    wordtoindex = json_data['wordtoindex']
    indextoword = json_data['indextoword']
    max_length = json_data['max_length']

    def classify(file_path):
        global label_packed
        img = encode(file_path, model_InceptionV3).reshape((1, 2048))
        label.configure(text=beam_search_predictions(img, max_length, wordtoindex, indextoword, model))

    def show_classify_button(file_path):
        classify_b = Button(top, text="识别", command=lambda: classify(file_path), padx=10, pady=5)
        classify_b.place(relx=0.79, rely=0.46)


    def upload_image():
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)
            original_image.configure(image=im)
            original_image.image = im
            label.configure(text='')
            show_classify_button(file_path)
        except:
            pass

    upload = Button(top, text="上传图像", command=upload_image, padx=10, pady=5)
    upload.pack(side=BOTTOM, pady=50)
    original_image.pack(side=BOTTOM, expand=True)
    label.pack(side=BOTTOM, expand=True)
    heading = Label(top, text="图文生成", pady=20)
    heading.pack()
    top.mainloop()
