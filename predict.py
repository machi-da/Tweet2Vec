import argparse
import configparser
import os
import glob
import logging
from logging import getLogger

import dataset
from model import Tweet2vec

import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--interval', '-i', type=int, default=100)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_dir = args.model_dir
    """LOAD CONFIG FILE"""
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(config_files) == 1, 'Put only one config file in the directory'
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = model_dir + 'log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Training start')
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    dropout = float(config['Parameter']['dropout'])
    """TRINING DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    interval = args.interval
    model_file = args.model
    """DATASET"""
    predict_data_file = config['Dataset']['predict_data_file']
    text, label = dataset.load_none_label(predict_data_file)

    vocab = dataset.Vocab()
    vocab.load(model_dir + 'vocab.pkl')
    vocab_size = len(vocab.vocab)

    label_dic = dataset.Label()
    label_dic.load(model_dir + 'label.pkl')
    label_size = len(label_dic.dic)

    iter = dataset.Iterator(text, label, vocab, label_dic, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = Tweet2vec(vocab_size, embed_size, hidden_size, dropout, label_size)
    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """PREDICT"""
    print(text)
    res = []
    for batch in iter():
        batch = dataset.converter(batch, gpu_id)
        proj_label = model.predict(*batch)
        print(proj_label)
        for p in proj_label:
            res.append(label_dic.reverse_dic[p])
    with open(predict_data_file + '.res', 'w')as f:
        [f.write('{}\t{}\n'.format(r, t)) for r, t in zip(res, text)]


if __name__ == '__main__':
    main()
