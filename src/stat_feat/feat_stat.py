from pyspark import SparkContext, SparkConf
import sys

conf = SparkConf().setAppName("test")#.setMaster("yarn-cluster")
sc = SparkContext(conf=conf)

#testFile=sc.textFile("hdfs://hqz-ubuntu-master:9000/data/train_ins/part-02001.gz")

testFile=sc.textFile("hdfs://hqz-ubuntu-master:9000/data/train_ins/part-*.gz")
def fmap(line):
    segs = line.split(" ")

    ret = []

    for s in segs:
        s_segs = s.split(":")

        if len(s_segs) == 2:
            ret.append((int(s_segs[0]),(int(segs[1]),int(segs[2]))))

    return ret

def red(a,b):
    return (a[0]+b[0],a[1]+b[1])


res=testFile.flatMap(fmap).reduceByKey(red)

res.saveAsTextFile("hdfs://hqz-ubuntu-master:9000/data/feat_stat.3")
