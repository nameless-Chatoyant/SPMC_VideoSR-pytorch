import torch.utils.data as data_utils

class Data(data_utils.Dataset):
    def __init__(self, files):
        self.files = [i.strip() for i in open(files).readlines()]
    def __getitem__(self, idx):
        imgs = self.files[idx]
        return imgs
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    data = Data('data_train.txt')
    print(data.__len__())
    for i in range(data.__len__()):
        print(data.__getitem__(i))