#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: LayerSoftmax.py
# Date: 2016 Thu 28 Jul 2016 05:33:44 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from Layer import Layer

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class LayerSoftmax(Layer):
    def __init__(self, config):
        super(LayerSoftmax, self).__init__(config)
    
    def _forward(self, bottom_data):
        return self.softmax(bottom_data)
        return self.sigmoid(bottom_data)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

def main():
    conf = {
            "debug": True,
            }
    t = LayerSoftmax(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

