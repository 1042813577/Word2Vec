from gensim.models import Word2Vec
import warnings
from gensim.models.word2vec import LineSentence

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def train_w2v_model(path):
    # 要求path文件已经分好词，且以空格相隔
    # workers:多线程数    size：词向量大小    min_count:出现频率低于此值的词将被过滤
    w2v_model = Word2Vec(LineSentence(path), workers=4, size=100, min_count=2)
    # w2v_model.save('w2v.model') 可进行增量训练
    # 二进制保存，加载更快但不能增量训练
    w2v_model.wv.save('w2v.model')


def add_model(model_path, data_path, epoch: int):
    model = Word2Vec.load(model_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    # 二维
    new_words = []
    for line in lines:
        line = line.strip().split(' ')
        new_words.append(line)
    model.train(sentences=new_words, epochs=epoch, total_examples=len(new_words))
    model.save('w2v_new.model')


def get_model_from_file():
    model = Word2Vec.load('w2v.model')
    return model



