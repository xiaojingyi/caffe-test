#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Base.py
# Date: 2016 Wed 27 Jul 2016 08:05:17 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Base(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Base init")
        self.config = config
        self.debug = config["debug"]
        #super(Base, self).__init__(config)
    
    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = Base(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

