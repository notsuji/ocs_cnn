#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.links as L

from chainer.cuda import to_gpu
from chainer import training
from chainer.training import extensions
from original_models import VGG
from original_data import get_data


def main():

    parser = argparse.ArgumentParser(description='Chainer example:cfiar-VGG')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()

    with chainer.using_config('train', False):

        model = L.Classifier(VGG(10))

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

        chainer.serializers.load_npz('./result/model_epoch_10', model)

        predict = get_data(r"./predict")

        for predict_ in predict:

            x, t = predict_

            if args.gpu >= 0:
                x = to_gpu(x, 0)

            x = x[None, ...]
            y = model.predictor(x)
            y = y.data
            print('label:', t, ' predicted_label:', y.argmax(axis=1)[0])

if __name__ == '__main__':
    main()
