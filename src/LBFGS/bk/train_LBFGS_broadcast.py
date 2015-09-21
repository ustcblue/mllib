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


def randomize(ins,sampling_rate):
    if random.randint(1,int(1/sampling_rate)) == 1:
        return [ins]
    else:
        return []

def ins_gradient(ins,broadcast_weights,accum):
    grad = []
    accum.add(1)
    pred = ins.predict(broadcast_weights.value)
    
    for f in ins.feat:
        grad.append((f,(pred - ins.label)))
    
    return grad

def eval_ins_map(ins,broadcast_weights):
    return ( ins.predict(broadcast_weights.value), ins.label)


class LBFGS_optimization:
    s = []
    y = []
    p = []

    M = 100
    
    sc = None

    feat_weight = {}
    feat_dict = {}

    train_ins = None
    eval_ins = None

    train_ins_count = 0
    eval_ins_count = 0

    theta = 4
    learning_rate = 0
    SAMPLING_RATE = 0
    ITER_MAX = 0

    def __init__(self,context=None):

        if context == None:
            conf = SparkConf().setAppName("LR_SGD")
            context = SparkContext(conf=conf)

        self.sc = context
        
        self.learning_rate = 0.5
        self.ITER_MAX = 1000
        self.theta = 4
        SAMPLING_RATE = 1

    def get_Hk_gk(self,k,grad_vec):
        if k <= self.M:
            L = k
        else:
            L = self.M

        q = grad_vec

        alpha = [0] * L

        for i in range(L-1,-1,-1):

            res = self.s[i].T * q * self.p[i]
        
            if res.shape != (1,1):
                raise TypeError("dimension error for res")

            alpha[i] = res[0,0]

            q = q - self.y[i] * alpha[i]

        beta = [0]*L

        z = q

        for i in range(0,L):
        
            res = self.y[i].T * z * self.p[i]

            if res.shape != (1,1):
                raise TypeError("dimension error for res")

            beta[i] = res[0,0]

            z = z + self.s[i]*(alpha[i]-beta[i])

        return z

    def normalization2mat(self,ins_grad,ins_count):

        grad_mat = scipy.mat(np.zeros((len(feat_dict),1)))
        norm_mat = scipy.mat(np.zeros((len(feat_dict),1)))

        for f in ins_grad:
            feat_idx=self.feat_dict[f[0]]
            grad_mat[feat_idx,0] = f[1] / ins_count
        
        for f in self.feat_weight:
            feat_idx=self.feat_dict[f]
            norm_mat[feat_idx,0] = theta / ins_count * self.weight[f]
        
        return grad_mat + norm_mat

    def update_weight(self,step,weight):

        for f in self.feat_dict:
            feat_idx=self.feat_dict[f]
            v = step[feat_idx,0]

            if v != 0:
                if f not in weight:
                    weight[f] = 0.0

                weight[f] += v

    def line_search(self,mini_batch_ins,search_direction):

        min_loss = 99999999
        min_loss_step = 0
        
        steps = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
        #steps = [0.5]
        for step in steps:

            weight = copy.deepcopy(self.feat_weight)
        
            s = search_direction * step

            self.update_weight(s,weight)

            broadcast_weight = self.sc.broadcast(weight)

            eval_res = mini_batch_ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()

            [mae, loss] = utils.get_loss(eval_res,weight,self.theta)
        
            print "step: %f, loss:%f" % (step,loss)

            if loss <= min_loss:
                min_loss = loss
                min_loss_step = step
        
        return min_loss_step

    def line_search_backtracing(self,mini_batch_ins,search_direction):

        c = 0.25
        t = 0.8

        step = 1.0
        
        broadcast_weight = self.sc.broadcast(self.feat_weight)
        
        eval_res = mini_batch_ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()
        
        [mae, loss] = utils.get_loss(eval_res,feat_weight,self.theta)
        
        while True:
            
            weight = copy.deepcopy(self.feat_weight)
        
            s = search_direction * step

            update_weight(s,self.feat_dict,weight)

            broadcast_weight = self.sc.broadcast(weight)

            eval_res = ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()

            [mae_new, loss_new] = utils.get_loss(eval_res,weight,THETA)
        
            g2 = search_direction.T * search_direction 
            print "step: %f, loss: %f, org_loss: %f, g2: %f, step:%f" % (step, loss_new,loss,g2, step)

            if loss_new > loss + c * step * g2:
                step = step * t
            else:
                break
        
        return step

    def load_ins_feat(self,train_file,eval_file,feat_file):
        [self.train_ins,self.train_ins_count] = utils.load_ins(self.sc,train_file)
    
        [self.eval_ins,self.eval_ins_count] = utils.load_ins(self.sc,eval_file)
    
        self.feat_dict = utils.load_feat(self.sc,feat_file)

    def evaluate(self,instance):
        broadcast_feat = self.sc.broadcast(self.feat_weight)
        
        eval_res = instance.map(lambda ins: eval_ins_map(ins,broadcast_feat)).sortByKey().collect()

        return utils.get_eval_stat(eval_res,self.feat_weight,THETA)
        

    def mini_batch_gradient(self):
        
        broadcast_feat = self.sc.broadcast(self.feat_weight)
        
        accum = self.sc.accumulator(0)

        min_batch_ins = self.train_ins.flatMap(lambda ins: randomize(ins,self.SAMPLING_RATE)).cache()

        ins_grad = min_batch_ins.flatMap(lambda ins: calc_gradient(ins, broadcast_feat, accum)).reduceByKey(lambda a,b: a+b).collect()

        grad_vector = self.normalization2mat(ins_grad,accum.value)

        return [min_batch_ins,grad_vector]

    def train(self):

        self.load_ins_feat("hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-00051", \
                "hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*", \
                "hdfs://hqz-ubuntu-master:9000/data/filtered_feat/*")
        cur_iter = 0

        [min_batch_ins,grad_vector] = self.mini_batch_gradient()

        d = grad_vector * (-1)

        while cur_iter < ITER_MAX:
        
            print ( "iteration %d" % cur_iter )

            #step = learning_rate / (math.sqrt(cur_iter+1))

            #step = line_search(min_batch_ins,sc,feat_weight,feat_dict,d,THETA)
        
            step = self.line_search_backtracing(min_batch_ins,d)

            si = d * step
        
            print ("step: %f" % step)

            if len(s) >= M:
                s.pop(0)

            self.s.append(si)

            self.update_weight(si,self.feat_weight)

            [auc,mae,loss] = self.evaluate(min_batch_ins)
            print ("selected %d samples: train_set : auc :%f, mae: %f, loss: %f" % (accum.value,auc,mae,loss))
            [auc,mae,loss] = self.evaluate(eval_ins)
            print ("test_set: auc :%f, mae: %f,loss: %f" % (auc,mae,loss))

            utils.output(cur_iter, None, self.feat_weight,eval_res)
            
            [min_batch_ins,new_grad_vector] = self.mini_batch_gradient()

            yi = new_grad_vector - grad_vector

            if len(y) >= M:
                y.pop(0)
        
            y.append(yi)

            if len(p) >= M:
                p.pop(0)

            tmp = yi.T*si

            if tmp.shape != (1,1):
                raise TypeError("dimension error for tmp")

            p.append(1/(tmp[0,0]))

            grad_vector = new_grad_vector

            cur_iter += 1

            d =  get_Hk_gk(cur_iter,grad_vector)
        
if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_SGD")
    sc = SparkContext(conf=conf)

    lbfgs = LBFGS_optimization(sc)
    lbfgs.train()
