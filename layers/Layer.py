#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Layer.py
# Date: 2016 Wed 27 Jul 2016 08:05:31 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import gnumpy as gnp
import math
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Layer(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Layer init")
        self.config = config
        self.debug = config["debug"]
    
    def sigmoid(self, x):
        return 1 / (1 + math.e ** (-x))

    def setup(self):
        self._setup()
        return

    def step(self, bottom_data):
        top_data = self._forward(bottom_data)
        self._backward(top_data, bottom_data)
        return top_data

    def forward(self, bottom_data):
        return self._forward(bottom_data)

    def _forward(self, bottom_data):
        return bottom_data

    def _backward(self, top_data, bottom_data):
        return

def main():
    conf = {
            "debug": True,
            }
    t = Layer(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

