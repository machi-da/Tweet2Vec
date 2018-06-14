import argparse
import configparser
import os
import glob
import logging
from logging import getLogger

import dataset
from model import Tweet2vec

# os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--interval', '-i', type=int, default=10000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
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
    weightdecay = float(config['Parameter']['weightdecay'])
    gradclip = float(config['Parameter']['gradclip'])
    vocab_size = int(config['Parameter']['vocab_size'])
    label_size = int(config['Parameter']['label_size'])
    """TRINING DETAIL"""
    gpu_id = args.gpu
    n_epoch = args.epoch
    batch_size = args.batch
    interval = args.interval
    """DATASET"""
    train_data_file = config['Dataset']['train_data_file']
    valid_data_file = config['Dataset']['valid_data_file']
    test_data_file  = config['Dataset']['test_data_file']

    train_text, train_label = dataset.load(train_data_file)
    valid_text, valid_label = dataset.load(valid_data_file)
    test_text , test_label  = dataset.load(test_data_file)

    logger.info('train size: {}, valid size: {}, test size{}'.format(len(train_text), len(valid_text), len(test_text)))

    vocab = dataset.Vocab()
    vocab.build(train_text, vocab_size)
    dataset.save_pickle(model_dir + 'vocab.pkl', vocab.vocab)
    train_vocab_size = len(vocab.vocab)

    label_dic = dataset.Label()
    label_dic.build(train_label, label_size)
    dataset.save_pickle(model_dir + 'label.pkl', label_dic.dic)
    train_label_size = len(label_dic.dic)

    logger.info('vocab size: {}, label size: {}'.format(train_vocab_size, train_label_size))

    train_iter = dataset.Iterator(train_text, train_label, vocab, label_dic, batch_size, sort=True, shuffle=True)
    # train_iter = dataset.Iterator(train_text, train_label, vocab, label_dic, batch_size, sort=False, shuffle=False)
    valid_iter = dataset.Iterator(valid_text, valid_label, vocab, label_dic, batch_size, sort=False, shuffle=False)
    test_iter  = dataset.Iterator(test_text, test_label, vocab, label_dic, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = Tweet2vec(train_vocab_size, embed_size, hidden_size, dropout, train_label_size)
    """OPTIMIZER"""
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(weightdecay))
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TRAIN"""
    sum_loss = 0
    loss_dic = {}
    for epoch in range(1, n_epoch + 1):
        for i, batch in enumerate(train_iter(), start=1):
            batch = dataset.converter(batch, gpu_id)
            loss = optimizer.target(*batch)
            sum_loss += loss.data
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            if i % interval == 0:
                logger.info('E{} ## iteration:{}, loss:{}'.format(epoch, i, sum_loss))
                sum_loss = 0
        chainer.serializers.save_npz(model_dir + 'model_epoch{}.npz'.format(epoch), model)
        # chainer.serializers.save_npz(model_dir + 'optimizer_epoch{0}.npz'.format(epoch), optimizer)

        """EVALUATE"""
        valid_loss = 0
        for batch in valid_iter():
            batch = dataset.converter(batch, gpu_id)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                valid_loss += optimizer.target(*batch).data
        logger.info('E{} ## val loss:{}'.format(epoch, valid_loss))
        loss_dic[epoch] = valid_loss

        """TEST"""
        correct = 0
        total = 0
        for batch in test_iter():
            batch = dataset.converter(batch, gpu_id)
            proj_label = model.predict(*batch)
            for proj, gold in zip(proj_label, batch[1]):
                if proj == gold:
                    correct += 1
                total += 1
        logger.info('E{} ## test accuracy:{}'.format(epoch, correct / total))

    """MODEL SAVE"""
    best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
    logger.info('best_epoch:{}'.format(best_epoch))
    chainer.serializers.save_npz(model_dir + 'best_model.npz', model)


if __name__ == '__main__':
    main()
