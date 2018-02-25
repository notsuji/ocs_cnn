# coding:utf-8

import chainer
from PIL import Image
import os


def make_data(makedir):
    train, test = chainer.datasets.get_mnist(ndim=3)

    for dataset_dir in ["train", "test"]:
        # ディレクトリ作成
        for i in range(10):
            maked_dir = os.path.join(makedir, dataset_dir, str(i))
            if os.path.exists(maked_dir) is False:
                os.makedirs(maked_dir)

        # データの画像化
        data = train if dataset_dir == "train" else test
        for i, (x, t) in enumerate(data):
            img = Image.fromarray(x[0] * 255)
            img = img.convert("RGB")
            image_name = "{:>08}_{}.png".format(i, t)
            img.save(os.path.join(makedir, dataset_dir, str(t), image_name))


if __name__ == '__main__':
    make_data(makedir="./Data")