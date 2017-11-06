#!/usr/bin/python
import sys

logfile = sys.argv[1]
with open(logfile, "r") as rhead:
    txt = rhead.readlines()
    accum = 0
    for elem in txt:
        accum += float(elem.split(" ")[3][:-1])
    print accum / len(txt)
