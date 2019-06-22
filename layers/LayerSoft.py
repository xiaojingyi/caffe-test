#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: LayerSoft.py
# Date: 2016 Fri 29 Jul 2016 10:30:51 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import gnumpy as gnp
import numpy as np
import math
sys.path.append("/datas/lib/py")
from Layer import Layer

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class LayerSoft(Layer):
    def __init__(self, config):
        super(LayerSoft, self).__init__(config)
        self.setup()
    
    def _setup(self):
        num_bottom = self.config['num_bottom']
        num_top = self.config['num_top']
        self.out_ave = np.zeros((num_top))

        if self.config.has_key("drop_connect"):
            p = self.config["drop_connect"]
        else:
            p = 0
        self.mask = (
                np.random.random((num_bottom, num_top)) * 100 
                > 100 * p
                ).astype(np.int)

        if self.config.has_key("std_param"):
            std = self.config["std_param"]
        else:
            std = 0.01
        self.params = np.random.normal(0, std, (num_bottom, num_top))
        self.params = self.params * self.mask

        if self.config.has_key("std_lr"):
            std = self.config["std_lr"]
        else:
            std = 0.001
        self.params_lr = abs(np.random.normal(0, std, (num_bottom, num_top)))
        self.params_lr = self.params_lr * self.mask

        if self.config.has_key("threshold"):
            threshold = self.config["threshold"]
        else:
            threshold = 0.001
        self.threshold = abs(np.random.normal(0, threshold, (num_top)))

        if self.config.has_key("gpu"):
            self.out_ave = gnp.garray(self.out_ave)
            self.params = gnp.garray(self.params)
            self.params_lr = gnp.garray(self.params_lr)
            self.threshold = gnp.garray(self.threshold)
        self.i = 0
        return

    def _forward(self, bottom_data):
        self.param_diff = (bottom_data - self.params.transpose()).transpose()
        tmp = self.param_diff * self.mask
        d = np.where(tmp < 0)
        tmp = abs(tmp) ** 1.2
        tmp[d] *= -1
        top_data = tmp.sum(0) / (self.mask.sum(0) + 0.00001)

        self.avg_diff = top_data.copy()
        """

        top_data = top_data - self.out_ave
        d = np.where(top_data < 0)
        top_data = abs(top_data) ** 1.2
        top_data[d] *= -1

        top_data = (top_data - top_data.mean()) / (top_data.max() - top_data.min())
        """
        #print top_data
        return top_data

    def _backward(self, top_data, bottom_data):
        self.params = self.params * (1 - self.params_lr ** 2)
        tmp = self.params_lr * self.param_diff
        self.params = self.params + tmp

        self.out_ave = self.out_ave * (1 - self.threshold)
        self.out_ave = self.out_ave + self.threshold * self.avg_diff
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = LayerSoft(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

