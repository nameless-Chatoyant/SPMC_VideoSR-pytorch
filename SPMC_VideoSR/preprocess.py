import os
import random

def preprocess(cfg):
    records = [os.path.join(root, filename) for root, dirs, filenames in os.walk(cfg.dir) for filename in filenames if filename.endswith('.wav')]
    total_num = len(records)
    test_num = int(cfg.test_ratio * total_num)
    train_num = total_num - test_num
    train_records = records[0:train_num]
    test_records = records[train_num:]

    with open(cfg.name + '_all.txt', 'w') as f:
        f.writelines((i + '\n' for i in records))
    with open(cfg.name + '_train.txt', 'w') as f:
        f.writelines((i + '\n' for i in train_records))
    with open(cfg.name + '_test.txt', 'w') as f:
        f.writelines((i + '\n' for i in test_records))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='comma separated list of GPU(s) to use.', default = 'data')
    parser.add_argument('--test_ratio', help='ratio of test data', default = 0.1)
    parser.add_argument('--dir', help="directory of logging", default='./')
    args = parser.parse_args()
    preprocess(args)