import pickle
import numpy as np
from collections import Counter
from chainer.dataset.convert import to_device


def load(file_name):
    text = []
    label = []
    with open(file_name, 'r')as f:
        for line in f:
            line = line.strip().split('\t')
            label.append(line[0])
            text.append(line[1])
    return text, label


def load_none_label(file_name):
    text = []
    label = []
    with open(file_name, 'r')as f:
        for line in f:
            line = line.strip()
            text.append(line)
            label.append('')  # ダミーで空文字をappend
    return text, label


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def converter(batch, gpu_id=None):
    text =[to_device(gpu_id, b) for b in batch[0]]
    label = to_device(gpu_id, batch[1])
    return text, label


class Vocab:
    def __init__(self):
        self.vocab = None

    def load(self, vocab_file):
        self.vocab = load_pickle(vocab_file)

    def build(self, text, vocab_size=10000, freq=0):
        self.vocab = self._build_vocab(text, vocab_size, freq)

    def _build_vocab(self, text, vocab_size, freq):
        vocab = {'<unk>': 0}
        charactors_counter = Counter()
        charactors = []
        for i, sentence in enumerate(text):
            charactors.extend([c for c in sentence])
            # 10000ごとにCounterへ渡す
            if i % 10000 == 0:
                charactors_counter += Counter(charactors)
                charactors = []
        else:
            charactors_counter += Counter(charactors)

        for k, v in charactors_counter.most_common():
            if len(vocab) >= vocab_size:
                break
            if v <= freq:
                break
            vocab[k] = len(vocab)
        return vocab

    def char2id(self, sentence):
        vocab = self.vocab
        sentence_id = [vocab.get(c, vocab['<unk>']) for c in sentence]
        return np.array(sentence_id, dtype=np.int32)


class Label:
    def __init__(self):
        self.dic = None
        self.reverse_dic = None

    def load(self, label_dic_file):
        self.dic = load_pickle(label_dic_file)
        self.reverse_dic = self._set_reverse_dic(self.dic)

    def build(self, label, label_size=1000):
        self.dic = self._build_dic(label, label_size)
        self.reverse_dic = self._set_reverse_dic(self.dic)

    def _build_dic(self, label, label_size):
        dic = {'<unk>': 0}
        label_counter = Counter(label)

        for k, v in label_counter.most_common():
            if len(dic) >= label_size:
                break
            dic[k] = len(dic)
        return dic

    def _set_reverse_dic(self, dic):
        reverse_dic = {}
        for k, v in dic.items():
            reverse_dic[v] = k
        return reverse_dic

    def label2id(self, l):
        return np.array(self.dic.get(l, self.dic['<unk>']), dtype=np.int32)


class Iterator:
    def __init__(self, text, label, vocab, label_dic, batch_size, sort=True, shuffle=True):
        self.text = text
        self.label = label
        self.vocab = vocab
        self.label_dic = label_dic
        self.batch_size = batch_size
        self.sort = sort
        self.shuffle = shuffle

    def __call__(self):
        label = self.label
        text = self.text
        batch_size = self.batch_size

        t_data = []
        l_data = []
        for t, l in zip(text, label):
            t = self.vocab.char2id(t)
            t_data.append(t)

            l = self.label_dic.label2id(l)
            l_data.append(l)

        if self.sort:
            data = zip(t_data, l_data)
            data = sorted(data, key=lambda x: len(x[0]), reverse=True)
            t_data, l_data = zip(*data)
        t_batch = [t_data[b * batch_size: (b + 1) * batch_size] for b in range(int(len(t_data) / batch_size) + 1)]
        l_batch = [l_data[b * batch_size: (b + 1) * batch_size] for b in range(int(len(l_data) / batch_size) + 1)]

        if self.shuffle:
            data = list(zip(t_batch, l_batch))
            np.random.shuffle(data)
            t_batch, l_batch = zip(*data)

        for t, l in zip(t_batch, l_batch):
            # 補足: len(data) == batch_sizeのとき、batchesの最後に空listができてしまうための対策
            if not t:
                continue
            l = np.array(l, dtype=np.int32)
            yield (t, l)