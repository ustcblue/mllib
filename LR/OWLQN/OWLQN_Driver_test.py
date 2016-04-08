import os
import sys
import math

import scipy

sys.path.append("../utils")

import OWLQN
import instance
from instance import Instance

def calc_ins_gradient(ins,weights):
    grad = []
    pred = ins.predict(weights, 1)
    
    for f in ins.feat:
        grad.append((f,(pred - ins.label)))
    
    return grad

class OWLQN_Driver:
    M = 5

    feat_weight = []
    feat_dict = {}
    train_ins = []
    feat_weight = []

    l2weight = 0

    def __int__(self,l2=0):
        self.M = 5
        self.train_ins = []
        self.l2_weight = l2

    def load_ins_feat(self,ins_file,feat_file):

        for line in open(ins_file,"r"):
            self.train_ins.append(Instance(line))

        idx = 0
        for line in open(feat_file,"r"):
            segs = line.strip().split(" ")
            if int(segs[0]) >= 500:
                [sign,slot] = segs[1].split(":")
                self.feat_dict[eval(sign)] = idx
                idx += 1
            else:
                break

        self.feat_weight = [0] * len(self.feat_dict)
     
    def score(self, ins, weight):
        s = 0
        for f in ins.feat:
            if f in weight:
                s += weight[f]
        
        if ins.label == 0:
            s = -s

        return s
    
    def score2(self, ins, weight):
        s = 0
        for f in ins.feat:
            if f in weight:
                s += weight[f]
        return s
    
    def loss_gradient2(self, weight_array):
        
        loss = 0.0
        
        weight = {}

        gradient = [0] * len(self.feat_dict)
        
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if weight_array[idx,0] != 0:
                weight[feat] = weight_array[idx,0]

        for i in range(0,len(self.feat_dict)):
            loss += 0.5*weight_array[i,0]*weight_array[i,0]*self.l2weight
            gradient[i] = self.l2weight * weight_array[i,0]

        for ins in self.train_ins:
            s = self.score2(ins,weight)
            
            pred = 1.0 / ( 1.0 + math.exp(-s) )

            if ins.label == 1:
                insLoss = - math.log(pred)
            elif ins.label == 0:
                insLoss = - math.log(1-pred)
            else:
                continue

            loss += insLoss
            
            for w in ins.feat:
                if w in self.feat_dict:
                    gradient[self.feat_dict[w]] += (pred - ins.label)
        
        #print loss,gradient
        return [loss, scipy.mat(gradient).T]

    def loss_gradient(self, weight_array):
        
        loss = 1.0
        
        weight = {}

        gradient = [0] * len(self.feat_dict)
        
        for feat in self.feat_dict:
            idx = self.feat_dict[feat]
            if weight_array[idx,0] != 0:
                weight[feat] = weight_array[idx,0]

        for i in range(0,len(self.feat_dict)):
            loss += 0.5*weight_array[i,0]*weight_array[i,0]*self.l2weight
            gradient[i] = self.l2weight * weight_array[i,0]

        for ins in self.train_ins:
            s = self.score(ins,weight)
            if s < -30:
                insLoss = -s
                pred = 0
            elif s > 30:
                insLoss = 0
                pred = 1
            else:
                t = 1.0 + math.exp(-s)
                insLoss = math.log(t)
                pred = 1.0 / t

            loss += insLoss
            
            pp = 1.0 - pred
            if ins.label == 1:
                pp = pp * (-1)
            for w in ins.feat:
                if w in self.feat_dict:
                    gradient[self.feat_dict[w]] += pp
        
        return [loss, scipy.mat(gradient).T]

if __name__ == "__main__":
    
    driver = OWLQN_Driver()
    driver.load_ins_feat("format_trans/filtered_ins_100","format_trans/feat.stat.sort")

    n = len(driver.feat_dict)

    owlqn_instance = OWLQN.OWLQN(5, n, 4, driver.loss_gradient2 )
    x = [0] * n

    owlqn_instance.owlqn(x)
