from pyspark import SparkContext, SparkConf
import sys
import random
import math
from operator import add

def get_feat(ins):
    segs = ins.split(" ")
    
    ret = []
    for i in range(3,len(segs)):
        [feat_sign, slot] = segs[i].split(":")
        
        ret.append((feat_sign, 1))

    return ret

def stat(sc):
    
    ins = sc.textFile("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/*")

    feat_stat = ins.flatMap(lambda ins: get_feat(ins)).reduceByKey(add)

    res = feat_stat.collect()

    fp = open("feat_stat","w")

    for r in res:
        fp.write("%s\t%d\n" % (r[0],r[1]))
    
    fp.close()

if __name__ == "__main__":

    conf = SparkConf().setAppName("stat_feat")
    sc = SparkContext(conf=conf)

    stat(sc)

