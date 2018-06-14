import sys
from collections import Counter


def main(file_name):
    max_num = 20000
    min_num = 500

    hashtag = []
    text = []
    with open(file_name, 'r')as f:
        for line in f:
            h, t = line.strip().split('\t')
            hashtag.append(h)
            text.append(t)
    hashtag_counter = Counter(hashtag)
    hashtag_list = [k for k, v in hashtag_counter.most_common() if min_num <= v <= max_num]

    data = []
    for h, t in zip(hashtag, text):
        if h in hashtag_list:
            data.append('{}\t{}'.format(h, t))
    with open(file_name + '_f', 'w')as f:
        [f.write(d + '\n') for d in data]
    with open(file_name + '_hashtag', 'w')as f:
        print('Unique hashtag: {}'.format(len(hashtag_counter)), file=f)
        print('Valid  hashtag: {}'.format(len(hashtag_list)), file=f)
        print('Data size: {}'.format(len(data)), file=f)


args = sys.argv
file_name = args[1]
main(file_name)