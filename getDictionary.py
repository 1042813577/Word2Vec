from collections import defaultdict


def save_word_dict(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in vocab:
            word, index = line
            f.write('%s\t%d\n' % (word, index))


def read_data(path1, path2, path3):
    with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2, open(path3, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words += line.split(' ')
        for line in f2:
            words += line.split(' ')
        for line in f3:
            words += line.split(' ')
    return words


def build_dictionary(words, sort=True, min_count=0, lower=False):
    result = []
    # 按频次
    if sort:
        # defaultdict默认字典，为不存在于字典中的键赋予默认值
        dic = defaultdict(int)
        for word in words:
            if word.strip() != '':
                dic[word] += 1
    # 按dic的值以降序排序
    dic = sorted(dic.items(), key=lambda d:d[1], reverse=True)
    # 用enumerate加序号
    for i, pair in enumerate(dic):
        key = pair[0]
        if min_count and min_count > pair[1]:
            continue
        result.append(key)
    else:
        for i, item in enumerate(words):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]

    return vocab


if __name__ == '__main__':
    words = read_data('./data/train_x_seg.txt', './data/train_y_seg.txt', './data/test_x.txt')
    vocab = build_dictionary(words)
    save_word_dict(vocab, './data/dictionary.txt')

    with open('./data/dictionary.txt', 'r', encoding='utf-8') as f1, open('./data/summary.txt', 'w', encoding='utf-8') as f2:
        line = f1.readlines()
        for i in range(300):
            f2.write(line[i])
