import os
import sys
import re
import gzip
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter

import mojimoji
import copy
from xml.sax.saxutils import unescape


class Preprocess:
    def __init__(self):
        self.hangul = re.compile(u'[\uac00-\ud7af\u3200-\u321f\u3260-\u327f\u1100-\u11ff\u3130-\u318f\uffa0-\uffdf\ua960-\ua97f\ud7b0-\ud7ff]')  # ハングル
        self.symbol = re.compile(r'…')  # 末尾によく現れるやつ
        self.rt = re.compile(r'^(RT|rt) ')  # リツイート
        self.reply = re.compile(r'@\w+:*')  # リプライ
        self.url = re.compile(r'https?://[!-~]+')  # url
        self.newline = re.compile(r'\\n')  # 改行
        self.space = re.compile(r'( |　)+')  # 連続するスペース

        self.hashtag = re.compile(r'[#＃][Ａ-Ｚａ-ｚA-Za-z一-鿆0-9０-９ぁ-ヶｦ-ﾟー_]+')  # ハッシュタグ

    def __call__(self, sentence):
        if re.search(self.hangul, sentence):
            return ''
        sentence = mojimoji.zen_to_han(sentence, kana=False)
        sentence = sentence.lower()
        sentence = unescape(sentence)
        sentence = re.sub(self.symbol, '', sentence)
        sentence = re.sub(self.rt, '', sentence)
        sentence = re.sub(self.reply, '', sentence)
        sentence = re.sub(self.url, '', sentence)
        sentence = re.sub(self.newline, '', sentence)
        sentence = re.sub(self.space, ' ', sentence)
        sentence = sentence.strip()
        return sentence

    def extract_hashtag(self, sentence):
        sub_sentence = copy.copy(sentence)
        hashtag = []
        match = re.findall(self.hashtag, sentence)
        for m in match:
            sub_sentence = re.sub(m, '', sub_sentence)
            hashtag.append(m[1:])
        sub_sentence = re.sub(self.space, ' ', sub_sentence)
        sub_sentence = sub_sentence.strip()
        return sub_sentence, hashtag


def make_data(file_name):
    p_text = re.compile('"text":"(.*?)",')
    preprocess = Preprocess()
    data = []
    with gzip.open(file_name, 'rt', 'utf_8') as f:
        try:
            for line in f:
                try:
                    text = re.search(p_text, line).group(1)
                    text = preprocess(text)
                    sentence, hashtag = preprocess.extract_hashtag(text)
                    # sentence5文字以上, ハッシュタグ1以上3以下
                    if len(sentence) >= 5 and 1 <= len(hashtag) <= 3:
                        if sentence.isdigit():
                            continue
                        for h in hashtag:
                            # 1文字以下のhashtagはカット
                            if len(h) <= 1:
                                continue
                            data.append('{}\t{}'.format(h, sentence))
                except AttributeError:
                    pass
        except EOFError:
            pass
    return data


def main():
    args = sys.argv

    path = '/auto/local/data/corpora/tweet_JP/'
    # path = '/Users/machida/work/socialdog/Tweet2vec/'
    date = args[1]
    output = 'tweet{}.txt'.format(date)
    path_list = glob.glob(path + date)
    print('file num: {}'.format(len(path_list)))

    if os.path.isfile(output):
        print('File remove: {}'.format(output))
        os.remove(output)

    # 20個ずつの単位で並列処理
    n = 20
    chunk_path_list = [path_list[i:i+n] for i in range(0, len(path_list), n)]

    for paths in tqdm(chunk_path_list):
        size = 0
        data = Parallel(n_jobs=-1)([delayed(make_data)(p) for p in paths])
        with open(output, 'a')as f:
            for d in data:
                [f.write(dd + '\n') for dd in d]
                size += len(d)


if __name__ == '__main__':
    main()