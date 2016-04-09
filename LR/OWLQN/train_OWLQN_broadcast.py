from pyspark import SparkContext, SparkConf
import sys
import random
import math
import types
import copy
import scipy
import numpy as np

from OWLQN import OWLQN
sys.path.append("../utils/")
import utils

from utils import Instance

def calc_ins_gradient(ins,broadcast_weights):
    grad = []
    pred = ins.predict(broadcast_weights.value, 1)
    
    for f in ins.feat:
        grad.append((f,(pred - ins.label)))
    
    return grad

def eval_ins_map(ins,broadcast_weights):
    return ( ins.predict(broadcast_weights.value, 0), ins.label )


class OWLQN_train:

    M = 5
    
    sc = None

    feat_weight = []
    feat_dict = {}

    train_ins = None
    eval_ins = None

    mini_batch_ins = None
    mini_batch_size = 0

    train_ins_count = 0
    eval_ins_count = 0

    l1_weight = 0
    l2_weight = 0
    SAMPLING_RATE = 0.1

    def __init__(self, context = None, l1_weight = 4, l2_weight = 0):

        if context == None:
            conf = SparkConf().setAppName("LR_LBFGS")
            context = SparkContext(conf=conf)

        self.sc = context
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

        self.SAMPLING_RATE = 0.1

    def load_ins_feat(self,train_file,eval_file,feat_file):
        [self.train_ins,self.train_ins_count] = utils.load_ins(self.sc,train_file)
    
        [self.eval_ins,self.eval_ins_count] = utils.load_ins(self.sc,eval_file)
    
        self.feat_dict = utils.load_feat(self.sc,feat_file)
        self.feat_weight = [0.0] * len(self.feat_dict)

    def lossFunc_loss(self,x,broadcast_feat):

        eval_res = self.train_ins.map(lambda ins: eval_ins_map(ins,broadcast_feat)).sortByKey().collect()
        
        [auc,mae,ins_loss] = utils.get_eval_stat(eval_res)

        loss = ins_loss + self.l2_weight / 2 * ((x.T * x)[0,0])
        
        return loss

    def lossFunc_gradient(self,x,broadcast_feat):
        
        ins_grad = self.train_ins.flatMap(lambda ins: calc_ins_gradient(ins, broadcast_feat)).reduceByKey(lambda a,b: a+b).collect()
        
        grad_mat = scipy.mat(np.zeros((len(self.feat_dict),1)))

        for f in ins_grad:
            feat_idx=self.feat_dict[f[0]]
            grad_mat[feat_idx,0] = f[1]
        
        return grad_mat + self.l2_weight * x

    def output(self, iter, x):

        utils.output_weight(iter, self.feat_dict, x)

    def eval_func_for_train_set(self,x):
        
        weight_dict={}

        #translating feature matrix to a python dictionary
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if x[idx,0] != 0:
                weight_dict[feat] = x[idx,0]
        
        #broadcast the feature weight and calculate the gradient distributely
        broadcast_feat = self.sc.broadcast(weight_dict)

        loss = self.lossFunc_loss(x,broadcast_feat)
        loss_grad = self.lossFunc_gradient(x,broadcast_feat)
        
        return [loss, loss_grad]

    def eval_func_for_test_set(self,x):
        weight_dict={}

        #translating feature matrix to a python dictionary
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if x[idx,0] != 0:
                weight_dict[feat] = x[idx,0]

        #broadcast the feature weight and calculate the gradient distributely
        broadcast_feat = self.sc.broadcast(weight_dict)

        eval_res = self.eval_ins.map(lambda ins: eval_ins_map(ins,broadcast_feat)).sortByKey().collect()
        
        [auc,mae,ins_loss] = utils.get_eval_stat(eval_res)

        return [mae,auc]


    def train(self):

        #self.load_ins_feat("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-00051", \
        #self.load_ins_feat("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-*", \
        self.load_ins_feat("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-00051",
                "hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*", \
                "hdfs://hqz-ubuntu-master:9000/data/filtered_feat/*")

        feat_num = len(self.feat_dict)

        owlqn_instance = OWLQN(10,feat_num,self.l1_weight,self.eval_func_for_train_set,self.eval_func_for_test_set, self.output)

        owlqn_instance.owlqn(self.feat_weight)

if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_OWLQN")
    conf.set('spark.kryoserializer.buffer.max','512')
    sc = SparkContext(conf=conf)
    l1_weight = 0
    l2_weight = 4
    owlqn = OWLQN_train(sc,l1_weight,l2_weight)
    owlqn.train()
