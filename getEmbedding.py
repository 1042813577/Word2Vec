import random
import numpy as np
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from pathlib import Path


EMBEDDING_DIM = 2  # 词向量维度
PRINT_EVERY = 1000  # 绘图频率
EPOCHS = 1000  # 训练轮次
BATCH_SIZE = 15  # 每批数据大小
N_SAMPLES = 3  # 负样本大小
WINDOW_SIZE = 5  # 周边词窗口大小（skip-gram）
FREQ = 0  # 词汇出现频率
DELETE_WORDS = False  # 是否删除部分高频词
N_SAMPLES = 4  # 负样本大小


def get_freq(path):
    words_count = {}
    with open(path, 'r', encoding='utf-8') as f:
        i = 0
        for s in f:
            temp = s.split('\t')
            word = temp[0]
            count = int(temp[1].strip())
            words_count.update({word: count})
            if i == 0:
                print(word+' '+str(count))
            i += 1
    length = len(words_count)
    freqs = np.array(list(words_count.values())) / length
    words_freq = dict(zip(words_count.keys(), list(freqs)))
    return words_freq


def get_noise(words_freq):
    temp = np.array(list(words_freq.values())) ** 0.75
    temp2 = temp / np.sum(temp)
    result = dict(zip(words_freq.keys(), list(temp2)))
    return result


def get_target(words, idx):
    target_window = np.random.randint(1, WINDOW_SIZE + 1)  # 实际操作的时候，不一定会真的取窗口那么大小，而是取一个小于等于的随机数即可
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window if (idx + target_window) < len(words) else len(words) - 1
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])  # 切片含首不含尾
    return list(targets)


def get_batch(words):
    n_batches = len(words) // BATCH_SIZE
    words = words[:n_batches * BATCH_SIZE]
    for idx in range(0, len(words), BATCH_SIZE):
        batch_x, batch_y = [], []
        batch = words[idx:idx + BATCH_SIZE]
        for i in range(len(batch)):
            x = batch[i]
            y = get_target(batch, i)  # 最好让batch size要大于Windows size，否则Windows size没意义
            batch_x.extend([x] * len(y))  # 更加清晰skip gram的原理，不是一次性的一个输入，对应多个输出，而是一个个的对应，且输入不变
            batch_y.extend(y)
    yield batch_x, batch_y


def get_context(path1, path2, word_freqs):
    words = []
    with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2:
        for i in f1:
            words.extend(i.strip().split(' '))
        for j in f2:
            words.extend(j.strip().split(' '))
    if DELETE_WORDS:
        t = 1e-5
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in word_freqs.keys()}
        train_words = [w for w in words if random.random() < (1 - prob_drop[w])]
        return train_words
    else:
        return words


def build_w2v(sentences, size=5, min_count=5, iter=5):
    sentences = LineSentence(sentences)
    # sg=1表示使用skip gram，否则表示使用CBOW
    model = Word2Vec(sentences, size=size, min_count=min_count, iter=iter, sg=1)
    return model


def update_w2v(model_path, new_sentences_path):
    model = Word2Vec.load(model_path)
    sentences = []
    with open(new_sentences_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            sentences.append(line)
    model.train(sentences, total_examples=len(sentences), epochs=model.iter)
    return model


def build_ft(sentences, size=5, min_count=5, iter=5):
    sentences = LineSentence(sentences)
    model = FastText(sentences, size=size, min_count=min_count, iter=iter, sg=1)
    return model


def save_model(model, model_path):
    model.save(model_path)


def read_vocab(vocab_path):
    w2i = {}
    i2w = {}
    with open(vocab_path, encoding='utf-8') as f:
        i = 0
        for line in f:
            item = line.strip().split()
            try:
                w2i[item[0]] = i
                i2w[i] = item[0]
                i += 1
            except:
                print(line)
                continue
    return w2i, i2w


def build_embedding(vocab_path, model_path, model_type='Word2Vector'):
    # load model
    if model_type == 'Word2Vector':
        model = Word2Vec.load(model_path)
    elif model_type == 'FastText':
        model = FastText.load(model_path)
    print(type(model))

    # generage dict: index to vector; index is based on vocabulay
    w2i, _ = read_vocab(vocab_path)
    vocab_size = len(w2i)
    vector_size = model.vector_size
    embedding = {}
    count = 0
    for v, i in w2i.items():
        try:
            embedding[i] = model[v]
            count = count + 1
        except:
            embedding[i] = np.random.uniform(-0.25, 0.25, vector_size).astype(np.float32)

    print(f"Found {count}/{vocab_size} words in: {Path(model_path).name}")
    return embedding


def save_embedding(embedding, embedding_path):
    with open(embedding_path, 'w', encoding='utf-8') as f:
        for i, vector in embedding.items():
            s = str(i) + ' ' + ' '.join(map(str, vector.tolist()))+'\n'
            f.write(s)


if __name__ == "__main__":
    vocab_path = 'data/dictionary_freq.txt'
    word_count_path = 'data/dictionary_freq.txt'
    w2v_model_path = 'models/w2v.model'
    ft_model_path = 'models/ft.model'
    w2v_embedding_path = 'models/w2v_embed.txt'
    ft_embedding_path = 'models/ft_embed.txt'
    train_x_path = 'data/train_x_seg.txt'

    # word_freq = get_freq(word_count_path)
    # train_data = get_context('data/train_x_seg.txt', 'data/train_y_seg.txt', word_freq)
    # train model
    w2v_model = build_w2v(train_x_path)
    ft_model = build_ft(train_x_path)

    # save model
    save_model(w2v_model, w2v_model_path)
    save_model(ft_model, ft_model_path)

    # build embedding matrix
    w2v_embedding = build_embedding(vocab_path, w2v_model_path)
    ft_embedding = build_embedding(vocab_path, ft_model_path, 'FastText')

    # save embedding matrix
    save_embedding(w2v_embedding, w2v_embedding_path)
    save_embedding(ft_embedding, ft_embedding_path)