from pyspark import SparkContext, SparkConf
import sys
import random
import math

sys.path.append("utils/")
import utils

from utils import Instance

def calc_gradient(ins,weights,accum,sampling_rate):
        
    grad = []

    if random.randint(1,int(1/sampling_rate)) == 1:

        pred = ins.predict(weights.value)

        for i in range(0,len(ins.feat)):
            f = ins.feat[i]

            if f not in weights.value:
                continue

            g = [ 0, [] ]
            
            g[0] = pred - ins.label
            
            K = len(weights.value[f][1])

            if K == 0:
                pass
            else:
                for k in range(0,K):
                    
                    for p in range(0,len(ins.feat)):
                        if i != p:
                            if ins.feat[p] in weights.value and len(weights.value[ins.feat[p]][1]) > 0:
                                if len(g[1]) == 0:
                                    g[1] = [0.0] * K
                                g[1][k] += weights.value[ins.feat[p]][1][k]
                
                    if len(g[1]) > 0:
                        g[1][k] *= (pred - ins.label)

            grad.append((f,g))
        
        accum.add(1)

    return grad

def update_weight(grad,feat_weight, train_ins, ins_num, learning_rate,theta):

    auc_array = []

    multiply = 1.0

    g = {}

    for p in grad:
        
        feat_sign = p[0]
        feat_grad = p[1]

        k = len(feat_grad[1])

        g[feat_sign] = [ 0,[] ]
        
        g[feat_sign][0] = ( learning_rate*1.0 / ins_num * (feat_grad[0] - theta * feat_weight[feat_sign][0]) )

        for i in range(0,k):
            g[feat_sign][1].append( learning_rate*1.0 / ins_num * (feat_grad[1][i] - theta * feat_weight[feat_sign][1][i]) )

    feat_weight_last = {}

    for i in range(1,100):
        
        feat_weight_tmp = {}

        for feat_sign in g:
            
            k = len(g[feat_sign][1])

            feat_weight_tmp[feat_sign] = [ 0,[] ]
            
            if feat_sign not in feat_weight:
                feat_weight_tmp[feat_sign][0] = - g[feat_sign][0] * multiply
                for ii in range(0,k):
                    feat_weight_tmp[feat_sign][1].append( - g[feat_sign][1][ii] * multiply )
            else:
                feat_weight_tmp[feat_sign][0] = feat_weight[feat_sign][0] - g[feat_sign][0] * multiply
                for ii in range(0,k):
                    feat_weight_tmp[feat_sign][1].append(feat_weight[feat_sign][1][ii] - g[feat_sign][1][ii] * multiply)
     
        eval_res = train_ins.map(lambda ins: ( ins.predict(feat_weight_tmp), ins.label)).sortByKey().collect()

        [auc, mae, loss] = utils.get_eval_stat(eval_res)
        
        print "searching step: multiply %d: train auc: %f, train_loss:%f, train_mae: %f" % (multiply, auc,loss,mae)

        auc = -loss
        if len(auc_array) > 0 and auc <= auc_array[-1]:
            break
        
        auc_array.append(auc)
        feat_weight_last = feat_weight_tmp.copy()

        multiply = (i+1)*1.0
        
    return feat_weight_last
    
def add_gradient(a,b):
    c = [ 0, [] ]
    
    c[0] = a[0] + b[0]

    if len(a[1]) == 0:
        c[1] = b[1]
    elif len(b[1]) == 0:
        c[1] = a[1]
    else:
        for i in range(0,len(a[1])):
            c[1].append(a[1][i]+b[1][i])

    return c

def train(sc):
    
    feat_weight = {}

    learning_rate = 1
    ITER_MAX = 1000
    THETA = 4
    K = 8

    SAMPLING_RATE = 0.1

    [train_ins,train_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/*51")
    #[train_ins,train_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train.test/train.test")
    
    [eval_ins,eval_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*")
    #[eval_ins,eval_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval.test/eval.test")
    
    #feat_dict = utils.load_feat(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_feat/*")
    [ feat_dict, feat_freq ] = utils.load_feat_2(sc,"hdfs://hqz-ubuntu-master:9000/data/feat_count/*",10000)
    
    for f in feat_dict:
        feat_weight[f] = [ 0.0, [] ]
        
        if False: #feat_freq[f] >= 0:
            for i in range(0,K):
                feat_weight[f][1].append(random.uniform(0,0.001))
                #feat_weight[f][1].append(0)

    cur_iter = 0
    while cur_iter < ITER_MAX:
        print "============================================================================="
        print ( "iteration %d" % cur_iter )
        
        print "broadcasting feat_weight"
        broadcast_feat = sc.broadcast(feat_weight)

        accum = sc.accumulator(0)
        print "calculating gradient"
        grad = train_ins.flatMap(lambda ins: calc_gradient(ins, broadcast_feat,accum, SAMPLING_RATE)).reduceByKey(add_gradient).collect()

        print "updating feat_weight"
        feat_weight = update_weight(grad,feat_weight,train_ins,accum.value,learning_rate, THETA)

        #print "returned weight:"
        fp=open("weights_%d" % cur_iter,"w")
        for f in feat_weight:
            fp.write("%d\t%f\t%s\n" % (f, feat_weight[f][0],"\t".join([str(i) for i in feat_weight[f][1]])) )
        fp.close()
        
        print "evaluating..."
        eval_res = eval_ins.map(lambda ins: ( ins.predict(feat_weight), ins.label)).sortByKey().collect()

        print "getting eval res"
        [auc, mae, loss] = utils.get_eval_stat(eval_res)
        
        #utils.output(cur_iter, None, feat_weight,eval_res)
        
        print ("selected %d samples: auc :%f, mae: %f" % (accum.value,auc,mae))

        cur_iter += 1

if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_SGD")
    sc = SparkContext(conf=conf)

    train(sc)
