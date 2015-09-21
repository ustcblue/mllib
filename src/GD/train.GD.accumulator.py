from pyspark import SparkContext, SparkConf
import sys
import random

sys.path.append("../utils/")
import utils

from utils import Instance
from utils import WeightAccumulatorParam

def calc_gradient(ins, feat_weight_for_update , feat_weight_for_read,learning_rate,sampling_rate, theta):
    
    if random.randint(1,int(1/sampling_rate)) == 1:

        feat_value = feat_weight_for_read.value
        
        pred = ins.predict(feat_value)

        update_weight = {}

        for f in ins.feat:
            
            if f in feat_value:
                w = feat_value[f]
            else:
                w = 0

            update_weight[f] = - (pred - ins.label - theta * w )*learning_rate 

        feat_weight_for_update.add(update_weight)
        
        return 1
    else:
        return 0

def train(sc):
    
    learning_rate = 0.5
    ITER_MAX = 1000
    THETA = 4
    SAMPLING_RATE = 0.01

    [train_ins,train_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/*51")
    
    [eval_ins,eval_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*")

    cur_iter = 0
     
    single_sample_learning_rate = learning_rate / ( train_ins_count * SAMPLING_RATE )
    single_sample_theta = THETA/(train_ins_count * SAMPLING_RATE)

    print "single sample learning rate: %f " % single_sample_learning_rate
    print "single sample theta: %f" % single_sample_theta

    feat_weight = sc.accumulator({},WeightAccumulatorParam())
    
    broadcast_feat = sc.broadcast(feat_weight.value)

    while cur_iter < ITER_MAX:
        
        print ( "iteration %d" % cur_iter )
        
        selected_sample = train_ins.map(lambda ins: calc_gradient(ins, feat_weight,broadcast_feat,single_sample_learning_rate, SAMPLING_RATE, single_sample_theta)).reduce(lambda a,b: a+b)

        broadcast_feat = sc.broadcast(feat_weight.value)

        eval_res = eval_ins.map(lambda ins:utils.evalulate_map(ins,broadcast_feat) ).sortByKey().collect()

        [auc, mae, loss] = utils.get_eval_stat(eval_res)
        
        #utils.output(cur_iter, None, broadcast_feat.value,eval_res)
        
        print ("selected %d samples: auc :%f, mae: %f" % (selected_sample,auc,mae))

        cur_iter += 1


conf = SparkConf().setAppName("LR_SGD")
sc = SparkContext(conf=conf)

train(sc)
