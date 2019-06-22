#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: LayerFc.py
# Date: 2016 Wed 27 Jul 2016 08:05:31 PM CST
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

class LayerFc(Layer):
    def __init__(self, config):
        super(LayerFc, self).__init__(config)
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
            threshold = 0.1
        self.threshold = abs(np.random.normal(0, threshold, (num_top)))

        if self.config.has_key("gpu"):
            self.out_ave = gnp.garray(self.out_ave)
            self.params = gnp.garray(self.params)
            self.params_lr = gnp.garray(self.params_lr)
            self.threshold = gnp.garray(self.threshold)
        self.i = 0
        return

    def _backward(self, top_data, bottom_data):
        self.out_ave = (top_data - self.out_ave) / self.i + self.out_ave
        #print "ave", self.out_ave

        #self.params = self.params * (1 + self.params_lr)
        b_max = bottom_data.max()
        b_min = bottom_data.min()
        if b_max - b_min <= 0:
            bottom_norm = 0
        else:
            """
            bottom_norm =  -bottom_data / \
                    (abs(b_max) if abs(b_max) > abs(b_min) else abs(b_min))
            """
            bottom_norm =  -(bottom_data - bottom_data.mean()) \
                    / (b_max - b_min)
        if self.config.has_key("gpu"):
            tmp = gnp.zeros(self.params_lr.shape)
            tmp = tmp.transpose() + bottom_norm
        else:
            tmp = bottom_norm

        self.params = self.params * \
                ( 1 - 
                        self.params_lr * 
                        self.params_lr
                        )
        self.params = (
                self.params.transpose() +
                self.params_lr.transpose() * tmp
                ).transpose()
        self.params = self.params * self.mask
        """
        bottom_norm = bottom_data
        """
        return

    def _forward(self, bottom_data):
        params = self.params
        """
        params = self.params.copy()
        params = bottom_data - params.transpose()
        params = params.transpose()
        #print params
        params = (params - params.mean()) / (params.max() - params.mean())
        if self.config.has_key("gpu"):
            params = params.sigmoid()
        else:
            params = self.sigmoid(params)
        params = math.e ** params
        """
        out = bottom_data.dot(params) / self.mask.sum(0)
        #out = bottom_data.dot(params)# / bottom_data.shape[0]
        #out = out - self.out_ave * self.threshold
        #out = (out + abs(out)) / 2
        #out = math.e ** -abs(out - self.out_ave)
        #out = (out - out.mean()) / out.std()
        self.i += 1

        #print params
        #print out
        return out

def main():
    conf = {
            "debug": True,
            "is_train": True,
            "num_bottom": 300,
            "num_top": 2,
            "drop_connect": 0.8,
            }
    t = LayerFc(conf)
    for i in range(100000):
        tmp = np.random.random((300))
        #print tmp
        print t.forward(tmp)
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

