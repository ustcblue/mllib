from pyspark import SparkContext, SparkConf
import sys
import random
import math
import utils
import types
import copy
import scipy
import numpy as np
import LBFGS

from utils import Instance
from LBFGS import LBFGS

def calc_ins_gradient(ins,broadcast_weights):
    grad = []
    pred = ins.predict(broadcast_weights.value)
    
    for f in ins.feat:
        grad.append((f,(pred - ins.label)))
    
    return grad

def eval_ins_map(ins,broadcast_weights):
    return ( ins.predict(broadcast_weights.value), ins.label)


class LBFGS_train:

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

    theta = 4
    SAMPLING_RATE = 0.1

    def __init__(self,context=None):

        if context == None:
            conf = SparkConf().setAppName("LR_LBFGS")
            context = SparkContext(conf=conf)

        self.sc = context
        
        self.theta = 4
        self.SAMPLING_RATE = 0.1

    def load_ins_feat(self,train_file,eval_file,feat_file):
        [self.train_ins,self.train_ins_count] = utils.load_ins(self.sc,train_file)
    
        [self.eval_ins,self.eval_ins_count] = utils.load_ins(self.sc,eval_file)
    
        self.feat_dict = utils.load_feat(self.sc,feat_file)
        self.feat_weight = [0.0] * len(self.feat_dict)

    def lossFunc_loss(self,x):
        weight_dict={}

        #translating feature matrix to a python dictionary
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if x[idx,0] != 0:
                weight_dict[feat] = x[idx,0]

        #broadcast the feature weight and calculate the gradient distributely
        broadcast_feat = self.sc.broadcast(weight_dict)

        eval_res = self.train_ins.map(lambda ins: eval_ins_map(ins,broadcast_feat)).sortByKey().collect()
        
        [auc,mae,ins_loss] = utils.get_eval_stat(eval_res)

        loss = (ins_loss + self.theta / 2 * ((x.T * x)[0,0])) / self.train_ins_count
        
        return loss

    def lossFunc_gradient(self,x):
        
        weight_dict={}

        #translating feature matrix to a python dictionary
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if x[idx,0] != 0:
                weight_dict[feat] = x[idx,0]

        #broadcast the feature weight and calculate the gradient distributely
        broadcast_feat = self.sc.broadcast(weight_dict)
        
        ins_grad = self.train_ins.flatMap(lambda ins: calc_ins_gradient(ins, broadcast_feat)).reduceByKey(lambda a,b: a+b).collect()
        
        grad_mat = scipy.mat(np.zeros((len(self.feat_dict),1)))
        norm_mat = scipy.mat(np.zeros((len(self.feat_dict),1)))

        for f in ins_grad:
            feat_idx=self.feat_dict[f[0]]
            grad_mat[feat_idx,0] = f[1] / self.train_ins_count
        
        return grad_mat + self.theta / self.train_ins_count * x

    def eval_func_for_train_set(self,x):

        return [self.lossFunc_loss(x),self.lossFunc_gradient(x)]

    def select_mini_batch(self):
        
        return
        random_max = int(1/self.SAMPLING_RATE)
        
        self.mini_batch_ins = self.train_ins.flatMap(lambda ins: [ins] if random.randint(1,random_max) == 1 else [] ).cache()
        
        self.mini_batch_size = self.mini_batch_ins.count()

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
        self.load_ins_feat("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-*", \
                "hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*", \
                "hdfs://hqz-ubuntu-master:9000/data/filtered_feat/*")

        feat_num = len(self.feat_dict)

        lbfgs_instance = LBFGS(5,feat_num)

        lbfgs_instance.lbfgs(self.feat_weight,self.eval_func_for_train_set,self.eval_func_for_test_set)

if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_LBFGS")
    sc = SparkContext(conf=conf)

    lbfgs = LBFGS_train(sc)
    lbfgs.train()
