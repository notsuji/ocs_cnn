# -*- coding: utf-8 -*-
import os

import chainer
import chainer.links as L

from chainer.cuda import to_gpu
from chainer import training
from chainer.training import extensions
from original_models import VGG
from original_data import get_data

from sklearn import svm
from scipy import stats

def main():

    outliers_fraction = 0.05  # 全標本数のうち、異常データの割合

    # train
    with chainer.using_config('train', False):

        model1 = VGG(10)
        model2 = L.Classifier(model1)
        chainer.cuda.get_device_from_id(0).use()
        model2.to_gpu()

        print('Loading CNN model')
        chainer.serializers.load_npz('./result/model_epoch_20', model2)

        print('Loading Training dataset')
        # predict, _ = get_data(r"./data_0-5")
        predict, _ = get_data(r"/home/notsuji/chainer_src/origin/mnist_data/Data/train/0")

        print('Trainging OneClassSVN')
        clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')

        x3 = []

        for predict_ in predict:

            x, t = predict_

            x = to_gpu(x, 0)

            x = x[None, ...]
            _ = model2.predictor(x)
            x2 = chainer.cuda.to_cpu(model1.fc1_out.data)
            x3.append(x2[0])

        clf.fit(x3)

        # predict


        with chainer.using_config('train', False):

            model1 = VGG(10)
            model2 = L.Classifier(model1)

            chainer.cuda.get_device_from_id(0).use()
            model2.to_gpu()

            chainer.serializers.load_npz('./result/model_epoch_20', model2)

            print('Loading prediction dataset')
            predict, file_names = get_data(r"/home/notsuji/chainer_src/origin/mnist_data/Data/test")


            ok = 0
            ng = 0

            print('Predicting OneClassSVN')

            for predict_, file_name_ in zip(predict, file_names):

                x, t = predict_

                x = to_gpu(x, 0)

                x = x[None, ...]
                _ = model2.predictor(x)
                x2 = chainer.cuda.to_cpu(model1.fc1_out.data)

                y_pred = clf.decision_function(x2).ravel()  # 各データの超平面との距離、ravel()で配列を1D化
                threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)  # パーセンタイルで異常判定の閾値設定
                # y_pred = y_pred > threshold

                file_name_extless, _ = os.path.splitext(file_name_)
                y_pred = clf.predict(x2)

                if str(file_name_extless)[-1] != '0' and y_pred[0].astype(int) == 1:
                    ng += 1
                    print(file_name_, y_pred[0].astype(int))
                elif str(file_name_)[-1] == '0' and y_pred[0].astype(int) == -1:
                    ng += 1
                    print(file_name_, y_pred[0].astype(int))
                else:
                    ok += 1

            print("ok:", ok, "ng:", ng)


if __name__ == '__main__':
    main()
