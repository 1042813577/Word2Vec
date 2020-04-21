import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba


IGNORE_LIST = ['|', '[', ']', '图片', '语音', ' ']


def get_symbols(path):
    symbols = set()
    with open(path, 'r', encoding='utf-8') as f:
        for symbol in f:
            symbol = symbol.strip()
            symbols.add(symbol)
    return symbols


def remove_words(words_list):
    words_list = [word for word in words_list if word not in IGNORE_LIST]
    return words_list


def segment(sentence: str, cut_type='word', part_of_speech=False):
    if part_of_speech:
        if cut_type == 'word':
            # 句子参数
            words_poses = posseg.lcut(sentence)
            words, poses = [], []
            for w, p in words_poses:
                words.append(w)
                poses.append(p)
            return words, poses
        elif cut_type == 'char':
            words = list(sentence)
            poses = []
            for w in words:
                # 字符参数，得到只含1个元素的list，该元素为1个pair：(word/flag)，word=w，flag=w的词性
                w_p = posseg.lcut(w)
                poses.append(w_p[0].flag)
            return words, poses
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        else:
            return list(sentence)


def parse_data(train_path, test_path):
    # 读取csv文件
    train_data = pd.read_csv(train_path, encoding='utf-8')
    # 去除Report字段为na的行，inplace指定是否在值上操作
    train_data.dropna(subset=['Report'], how='any', inplace=True)
    # 其他字段若为na则填充为空串''
    train_data.fillna('', inplace=True)
    # 此处选择question与dialogue两个字段，拼接后作为x
    print(type(train_data.Question))
    train_x = train_data.Question.str.cat(train_data.Dialogue)
    train_y = []
    if 'Report' in train_data.columns:
        train_y = train_data.Report
        assert len(train_x) == len(train_y)

    test_data = pd.read_csv(test_path, encoding='utf-8')
    test_data.fillna('', inplace=True)
    test_x = test_data.Question.str.cat(test_data.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y


def save_data(train_x, train_y, test_x, path1, path2, path3, symbols_path):
    symbols = get_symbols(symbols_path)
    with open(path1, 'w', encoding='utf-8') as f1:
        count_1 = 0
        for line in train_x:
            if isinstance(line, str):
                words_seq = segment(line)
                words_seq = remove_words(words_seq)
                words_seq = [word for word in words_seq if word not in symbols]
                if len(words_seq) > 0:
                    result_line = ' '.join(words_seq)
                    f1.write('%s' % result_line)
                    f1.write('\n')
                    count_1 += 1
        print('train_x len=', count_1)

    with open(path2, 'w', encoding='utf-8') as f2:
        count_2 = 0
        for line in train_y:
            if isinstance(line, str):
                words_seq = segment(line)
                words_seq = remove_words(words_seq)
                words_seq = [word for word in words_seq if word not in symbols]
                if len(words_seq) > 0:
                    result_line = ' '.join(words_seq)
                    f2.write('%s' % result_line)
                    f2.write('\n')
                    count_2 += 1
                else:
                    # 如果去除特殊符号后啥也不剩了，则将“随时 联系”作为y
                    f2.write('随时 联系')
                    f2.write('\n')
                    count_2 += 1
        print('train_y len=', count_2)

    with open(path3, 'w', encoding='utf-8') as f3:
        count_3 = 0
        for line in test_x:
            if isinstance(line, str):
                words_seq = segment(line)
                words_seq = remove_words(words_seq)
                words_seq = [word for word in words_seq if word not in symbols]
                if len(words_seq) > 0:
                    result_line = ' '.join(words_seq)
                    f3.write('%s' % result_line)
                    f3.write('\n')
                    count_3 += 1
        print('test_x len=', count_3)


if __name__ == '__main__':
    train_x, train_y, test_x, _ = parse_data('./data/AutoMaster_TrainSet.csv', './data/AutoMaster_TestSet.csv')
    save_data(train_x, train_y, test_x, './data/train_x_seg.txt', './data/train_y_seg.txt', './data/test_x.txt', './symbols.txt')