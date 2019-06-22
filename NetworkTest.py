#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: NetworkTest.py
# Date: 2016 Thu 28 Jul 2016 11:50:57 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import sklearn
import sklearn.datasets
import sklearn.linear_model
import numpy as np
import gnumpy as gnp
import h5py
sys.path.append("/datas/lib/py")
from Network import Network

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class NetworkTest(Network):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: NetworkTest init")
        super(NetworkTest, self).__init__(config)

    def net(self, data_len):
        num = 200
        drop = 0.5
        self.define([
            ["LayerSoft", {
                "num_top": num, "num_bottom": data_len, 
                "drop_connect": drop,
                }],
            #["LayerFc", { "num_top": num, "num_bottom": num, "drop_connect": drop, }],
            #["LayerFc", { "num_top": num, "num_bottom": num, "drop_connect": drop, }],
            ["LayerSoft", {
                "num_top": 5, "num_bottom": num,
                "drop_connect": drop,
                }],
            #["LayerSoftmax", {}],
            ])
        return

    def mkData(self):
        X, y = sklearn.datasets.make_classification(
            n_samples=50000, 
            n_features=10, 
            n_redundant=0, 
            n_informative=3, 
            n_classes=3,
            n_clusters_per_class=1, hypercube=False, random_state=0
            )
        X = (X - X.mean(0)) / (X.max(0) - X.min(0))
        X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y, test_size=0.1)
        print X.shape
        print X.max(), X.min()
        return X, y, Xt, yt

    def testByliner(self, X, y, Xt, yt):
        clf = sklearn.linear_model.SGDClassifier(
            loss='log', n_iter=200, penalty='l2', alpha=1e-5, class_weight='auto')

        clf.fit(X, y)
        yt_pred = clf.predict(Xt)
        print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))
        return

    def train(self, X):
        epoch_len = 5
        for i in range(epoch_len):
            print "epoch", i
            j = 0
            for one in X:
                self.step(one)
                j += 1
                if j % (X.shape[0]/ 10) == 0:
                    pass
                    #print j

    def saveH5(self, X, y, Xt, yt, suffix):
        fname = "train_" + suffix + ".txt"
        dbname = "train_" + suffix + ".h5"
        with open(fname, 'w') as f:
            f.write(dbname + '\n')
        with h5py.File(dbname, 'w') as f:
            f['data'] = X.reshape((len(X), 1, 1, len(X[0])))
            f['label'] = y.astype(np.float32)

        fname = "test_" + suffix + ".txt"
        dbname = "test_" + suffix + ".h5"
        with open(fname, 'w') as f:
            f.write(dbname + '\n')
        with h5py.File(dbname, 'w') as f:
            f['data'] = Xt.reshape((len(Xt), 1, 1, len(Xt[0])))
            f['label'] = yt.astype(np.float32)
        return

    def transX(self, X):
        res = []
        for one in X:
            res.append(self.forward(one))
        return np.array(res)

    def run(self):
        X, y, Xt, yt = self.mkData()
        self.saveH5(X, y, Xt, yt, "pri")
        self.net(X.shape[1])
        print X
        self.testByliner(X, y, Xt, yt)
        self.train(X)

        X_ = (X - X.mean(0)) / X.std(0)
        Xt_ = (Xt - X.mean(0)) / X.std(0)
        print X
        self.testByliner(X_, y, Xt_, yt)

        X = self.transX(X)
        Xt = self.transX(Xt)
        print X.shape
        print X
        self.saveH5(X, y, Xt, yt, "after")
        self.testByliner(X, y, Xt, yt)

        """
        """
        X = (X - X.mean(0)) / X.std(0)
        Xt = (Xt - X.mean(0)) / X.std(0)
        print X
        self.testByliner(X, y, Xt, yt)
        return
    
def main():
    conf = {
            "debug": True,
            #"gpu": True,
            }
    t = NetworkTest(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

