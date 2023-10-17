import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


datasets = [
    'bicubic_train.npy',
    'bicubic_validation.npy',
    'data_train.npy',
    'data_validation.npy',
    'labels_train.npy',
    'labels_validation.npy'
]


model_weights = [
    '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
    'HAN_BIX2.pt',
    'HAT_SRx2_ImageNet-pretrain.pth',
    'IPT_sr2.pt'
]


checkpoints = [
    'ckpt-x2.pt',
    'RLSR-x2.pt'
]


def download_dataset(d, root='https://github.com/onebottlekick/RLSR/releases/download/dataset/'):
    for data in tqdm(d, desc='dataset'):
        download_url(root+data, '../dataset/'+data)
        
        
def download_model_weights(d, root='https://github.com/onebottlekick/RLSR/releases/download/model_weights/'):
    for data in tqdm(d, desc='model_weights'):
        download_url(root+data, '../model_weights/'+data)
        

def download_checkpoint(d, root='https://github.com/onebottlekick/RLSR/releases/download/pretrained_weights/'):
    for data in tqdm(d, desc='checkpoint'):
        download_url(root+data, '../checkpoint/'+data)
        
        
if __name__ == '__main__':
    download_dataset(datasets)
    download_model_weights(model_weights)
    download_checkpoint(checkpoints)