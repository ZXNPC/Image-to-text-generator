import string
from sklearn.utils import shuffle
from tqdm import tqdm

def remove_punctuation(text_original):
    dicts = {i:'' for i in string.punctuation}
    punc_table = str.maketrans(dicts)
    text_no_punctuation = text_original.translate(punc_table)
    return (text_no_punctuation)


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return (text_len_more_than1)


def remove_numeric(text):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if isalpha:
            text_no_numeric += " " + word
    return (text_no_numeric)


def text_clean(text_original):
    '''
    对text_original进行清洗，去除标点符号、单字符和数字
    :param text_original: original text
    :return: cleaned text
    '''
    if type(text_original) == list:
        text = []
        for t in text_original:
            text.append(text_clean(t))
        return text

    elif type(text_original) == dict:
        text = {}
        qbar = tqdm(text_original.keys())
        for key in qbar:
            qbar.set_description('Clean text')
            text[key] = text_clean(text_original[key])
        return text

    else:
        text = text_original.lower()
        text = remove_punctuation(text)
        text = remove_single_character(text)
        text = remove_numeric(text)
        return (text)


def add_start_end(caption_original):
    '''
    为所有caption头尾分别加上startseq和endseq作为标识符
    :param caption: original caption
    :return: processed caption
    '''
    if type(caption_original) == list:
        caption = []
        for c in caption_original:
            caption.append(add_start_end(c))
        return caption

    elif type(caption_original) == dict:
        caption = {}
        for key in caption_original.keys():
            caption[key] = add_start_end(caption_original[key])
        return caption

    else:
        caption = 'startseq ' + caption_original + ' endseq'
        return caption

# word = {}
#     nsents = 0
#     for img_caption in all_img_captions:
#         nsents += 1
#         for w in img_caption.split(' '):
#             word[w] = word.get(w, 0) + 1
#     vocabulary = [w for w in word if word[w] >= 10]

def make_vocab(all_img_info, threshold=10):
    '''
    统计单词出现频次并返回频次不低于threshold的单词
    :param all_img_info: informations of all images
    :param threshold: lower limit of occurrences of words
    :return: 高频词词库
    '''
    word = {}
    qbar = tqdm(all_img_info.keys())
    for key in qbar:
        qbar.set_description('Make vocabulary table')
        for caption in all_img_info[key]:
            for w in caption.split():
                word[w] = word.get(w, 0) + 1
    vocab = [w for w in word if word[w] >= threshold]
    return vocab

def train_test_split(all_img_paths, test_size, random_state=False):
    all_img_paths = [img_path.replace('\\', '/') for img_path in all_img_paths]
    if random_state==True:
        all_img_paths = shuffle(all_img_paths)
    split_pos = int(len(all_img_paths) * (1-test_size))
    train_img = all_img_paths[:split_pos]
    test_img = all_img_paths[split_pos:]
    return train_img, test_img