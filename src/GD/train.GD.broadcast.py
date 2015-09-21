from pyspark import SparkContext, SparkConf
import sys
import random
import math
import utils

from utils import Instance

def calc_gradient(ins,weights,accum,sampling_rate):
        
    grad = []

    if random.randint(1,int(1/sampling_rate)) == 1:

        pred = ins.predict(weights.value)

        for f in ins.feat:
            grad.append((f,(pred - ins.label)))
        
        accum.add(1)

    return grad

def update_weight(grad,feat_weight,ins_num, learning_rate,theta):

    for p in grad:
        if p[0] not in feat_weight:
            feat_weight[p[0]] = 0.0

        feat_weight[p[0]] -= learning_rate / ins_num * ( p[1] - theta * feat_weight[p[0]] )

def train(sc):
    
    feat_weight = {}

    learning_rate = 0.5
    ITER_MAX = 1000
    THETA = 4
    SAMPLING_RATE = 0.01

    [train_ins,train_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/*")
    
    [eval_ins,eval_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*")

    cur_iter = 0
    while cur_iter < ITER_MAX:
        
        print ( "iteration %d" % cur_iter )
        
        broadcast_feat = sc.broadcast(feat_weight)

        accum = sc.accumulator(0)

        grad = train_ins.flatMap(lambda ins: calc_gradient(ins, broadcast_feat,accum, SAMPLING_RATE)).reduceByKey(lambda a,b: a+b).collect()

        update_weight(grad,feat_weight,accum.value,learning_rate, THETA)

        eval_res = eval_ins.map(lambda ins: ( ins.predict(feat_weight), ins.label)).sortByKey().collect()

        [auc, mae] = utils.get_eval_stat(eval_res)
        
        utils.output(cur_iter, None, feat_weight,eval_res)
        
        print ("selected %d samples: auc :%f, mae: %f" % (accum.value,auc,mae))

        cur_iter += 1

if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_SGD")
    sc = SparkContext(conf=conf)

    train(sc)
