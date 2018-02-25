#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L

class VGG(chainer.Chain):

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            
            self.l1_1 = L.Convolution2D(None, 64, 3, pad=1,nobias=True)
            self.b1_1 = L.BatchNormalization(64)
            self.l1_2 = L.Convolution2D(None, 64, 3, pad=1,nobias=True)
            self.b1_2 = L.BatchNormalization(64)
            
            self.l2_1 = L.Convolution2D(None, 128, 3, pad=1,nobias=True)
            self.b2_1 = L.BatchNormalization(128)
            self.l2_2 = L.Convolution2D(None, 128, 3, pad=1,nobias=True)
            self.b2_2 = L.BatchNormalization(128)
            
            self.l3_1 = L.Convolution2D(None, 256, 3, pad=1,nobias=True)
            self.b3_1 = L.BatchNormalization(256)
            self.l3_2 = L.Convolution2D(None, 256, 3, pad=1,nobias=True)
            self.b3_2 = L.BatchNormalization(256)
            self.l3_3 = L.Convolution2D(None, 256, 3, pad=1,nobias=True)
            self.b3_3 = L.BatchNormalization(256)
            
            self.l4_1 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b4_1 = L.BatchNormalization(512)
            self.l4_2 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b4_2 = L.BatchNormalization(512)
            self.l4_3 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b4_3 = L.BatchNormalization(512)
            
            self.l5_1 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b5_1 = L.BatchNormalization(512)
            self.l5_2 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b5_2 = L.BatchNormalization(512)
            self.l5_3 = L.Convolution2D(None, 512, 3, pad=1,nobias=True)
            self.b5_3 = L.BatchNormalization(512)

            # self.fc1 = L.Linear(None, 512, nobias=True)
            self.fc1 = L.Linear(None, 128, nobias=True)
            # self.bn_fc1 = L.BatchNormalization(512)
            self.bn_fc1 = L.BatchNormalization(128)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

            self.fc1_out = ''
            
    def __call__(self, x):

        # 64 channel blocks:
        
        h = self.l1_1(x)
        h = self.b1_1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.3)
     
        h = self.l1_2(h)
        h = self.b1_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.l2_1(h)
        h = self.b2_1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l2_2(h)
        h = self.b2_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.l3_1(h)
        h = self.b3_1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l3_2(h)
        h = self.b3_2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l3_3(h)
        h = self.b3_3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.l4_1(h)
        h = self.b4_1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l4_2(h)
        h = self.b4_2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l4_3(h)
        h = self.b4_3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.l5_1(h)
        h = self.b5_1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l5_2(h)
        h = self.b5_2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.4)

        h = self.l5_3(h)
        h = self.b5_3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        self.fc1_out = h
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)
