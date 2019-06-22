#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Network.py
# Date: 2016 Wed 27 Jul 2016 11:52:54 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import gnumpy as gnp
sys.path.append("/datas/lib/py")
from Base import Base

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Network(Base):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Network init")
        self.config = config
        self.debug = config["debug"]
        super(Network, self).__init__(config)
    
    def define(self, network):
        self.network = []
        for i in range(len(network)):
            one = network[i]
            mname = one[0]
            config = one[1]
            config["debug"] = self.debug
            config["is_train"] = True
            lib = __import__("layers." + mname)
            classes = getattr(lib, mname)
            model = getattr(classes, mname)
            if self.config.has_key("gpu"):
                config["gpu"] = True
            layer = model(config)
            self.network.append(layer)
        return

    def step(self, data):
        if self.config.has_key("gpu"):
            data = gnp.garray(data)

        for i in range(len(self.network)):
            data = self.network[i].step(data)

        if self.config.has_key("gpu"):
            data = data.as_numpy_array()
        return data

    def forward(self, data):
        if self.config.has_key("gpu"):
            data = gnp.garray(data)

        for i in range(len(self.network)):
            data = self.network[i].forward(data)

        if self.config.has_key("gpu"):
            data = data.as_numpy_array()
        return data

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = Network(conf)
    num = 300
    drop = 0.8
    t.define([
        ["LayerSoft", {"num_top": num, "num_bottom": 3, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ["LayerSoft", {"num_top": num, "num_bottom": num, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ["LayerSoft", {"num_top": num, "num_bottom": num, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ["LayerSoft", {"num_top": num, "num_bottom": num, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ["LayerSoft", {"num_top": num, "num_bottom": num, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ["LayerSoft", {"num_top": 2, "num_bottom": num, "drop_connect": drop,}],
        #["LayerSoftmax", {}],
        ])

    for i in range (10000):
        tmp = np.random.random((3))
        print "data", tmp
        print t.step(tmp)
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

