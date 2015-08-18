from pyspark import SparkContext, SparkConf
import sys

conf = SparkConf().setAppName("test")#.setMaster("yarn-cluster")
sc = SparkContext(conf=conf)

testFile=sc.textFile("hdfs://hqz-ubuntu-master:9000/data/feat_stat.3/*")

def fmap(line):

    val = eval(line)

    if val[1][0] >= 50:

        return set([val[0]])

    else:
        return set([])

def fred(a,b):
    for v in b:
        a.add(v)
    return a

feat_dict = testFile.map(fmap).reduce(fred)

feat_dict_rdd = sc.parallelize(feat_dict)

feat_dict_rdd.saveAsTextFile("hdfs://hqz-ubuntu-master:9000/data/filtered_feat")

trainFile=sc.textFile("hdfs://hqz-ubuntu-master:9000/data/train_ins/*")


FEAT_DICT = sc.broadcast(feat_dict)

def filter_ins(line):
    global FEAT_DICT
    
    line = line.strip()

    segs = line.split(" ")
    output = [segs[0],segs[1],segs[2]]

    for s in segs:
        s_segs = s.split(":")
        if len(s_segs) == 2:
            if eval(s_segs[0]) in FEAT_DICT.value:
                output.append(s)

    if len(output) > 3:
        return " ".join(output)
    else:
        return ""

def filter_empty(line):
    if len(line) > 0:
        return True
    else:
        return False

filtered_ins_rdd = trainFile.map(filter_ins).filter(filter_empty)

filtered_ins_rdd.saveAsTextFile("hdfs://hqz-ubuntu-master:9000/data/filtered_ins")

