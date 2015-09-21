from pyspark import SparkContext, SparkConf
import sys
import random
import math
import utils
import types
import copy

from utils import Instance

import scipy
import numpy as np

s = []
y = []
p = []

M = 100

def get_Hk_gk(k,grad_vec):
    global s
    global y
    global p
    global M

    if k <= M:
        L = k
    else:
        L = M

    q = grad_vec

    alpha = [0] * L

    for i in range(L-1,-1,-1):

        res = s[i].T * q * p[i]
        
        if res.shape != (1,1):
            raise TypeError("dimension error for res")

        alpha[i] = res[0,0]

        q = q - y[i] * alpha[i]

    Hk = y[L-1].T * s[L-1] / (y[L-1].T * y[L-1])

    beta = [0]*L
    z = q #* Hk

    for i in range(0,L):
        
        res = y[i].T * z * p[i]

        if res.shape != (1,1):
            raise TypeError("dimension error for res")

        beta[i] = res[0,0]

        z = z + s[i]*(alpha[i]-beta[i])

    return z

def normalization2mat(ins_grad,ins_count,weights,feat_dict,theta):

    grad_mat = scipy.mat(np.zeros((len(feat_dict),1)))
    norm_mat = scipy.mat(np.zeros((len(feat_dict),1)))

    for f in ins_grad:
        
        feat_idx=feat_dict[f[0]]

        grad_mat[feat_idx,0] = f[1] / ins_count

    for f in weights:
        
        feat_idx=feat_dict[f]

        norm_mat[feat_idx,0] = theta / ins_count * weights[f]

    return grad_mat + norm_mat

def randomize(ins,sampling_rate):
    
    if random.randint(1,int(1/sampling_rate)) == 1:
        return [ins]
    else:
        return []

def calc_gradient(ins,weights,accum):
        
    grad = []

    accum.add(1)

    pred = ins.predict(weights.value)

    for f in ins.feat:
        grad.append((f,(pred - ins.label)))
        
    return grad

def eval_ins_map(ins,weights):
    return ( ins.predict(weights.value), ins.label)

def update_weight(step,feat_dict,feat_weight):

    for f in feat_dict:
        feat_idx=feat_dict[f]
        v = step[feat_idx,0]

        if v != 0:
            if f not in feat_weight:
                feat_weight[f] = 0.0

            feat_weight[f] -= v

def line_search(ins,sc,feat_weight,feat_dict,search_direction,THETA):

    min_loss = 99999999
    min_loss_step = 0

    steps = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    #steps = [0.5]
    for step in steps:

        weight = copy.deepcopy(feat_weight)
        
        s = search_direction * step

        update_weight(s,feat_dict,weight)

        broadcast_weight = sc.broadcast(weight)

        eval_res = ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()

        [mae, loss] = utils.get_loss(eval_res,weight, feat_dict, THETA)
        
        print "step: %f, loss:%f" % (step,loss)

        if loss <= min_loss:
            min_loss = loss
            min_loss_step = step

        #else:
        #    break


    return min_loss_step

def line_search_backtracing(ins,sc,feat_weight,feat_dict,search_direction,THETA):

    
    c = 0.25
    t = 0.8
    
    step = 1.0

    broadcast_weight = sc.broadcast(feat_weight)

    eval_res = ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()

    [mae, loss] = utils.get_loss(eval_res,feat_weight,feat_dict,THETA)
     
    while True:

        weight = copy.deepcopy(feat_weight)
        
        s = search_direction * step

        update_weight(s,feat_dict,weight)

        broadcast_weight = sc.broadcast(weight)

        eval_res = ins.map(lambda _ins_: eval_ins_map(_ins_,broadcast_weight)).collect()

        [mae_new, loss_new] = utils.get_loss(eval_res,weight,feat_dict,THETA)
        
        g2 = search_direction.T * search_direction 
        print "step: %f, loss: %f, org_loss: %f, g2: %f, step:%f" % (step, loss_new,loss,g2, step)

        if loss_new > loss + c * step * g2:
            step = step * t
        else:
            break


    return step

def mini_batch_gradient(sc,train_ins,feat_weight,feat_dict,THETA,SAMPLING_RATE):
    accum = sc.accumulator(0)
    
    broadcast_feat = sc.broadcast(feat_weight)
    
    min_batch_ins = train_ins.flatMap(lambda ins: randomize(ins,SAMPLING_RATE)).cache()

    ins_grad = min_batch_ins.flatMap(lambda ins: calc_gradient(ins, broadcast_feat, accum)).reduceByKey(lambda a,b: a+b).collect()

    new_grad_vector = normalization2mat(ins_grad,accum.value,feat_weight,feat_dict,THETA)

    return [min_batch_ins, new_grad_vector,accum.value]

def evaluate(sc,batch_ins,feat_weight,feat_dict,THETA):

    broadcast_feat = sc.broadcast(feat_weight)
        
    eval_res = batch_ins.map(lambda ins: eval_ins_map(ins,broadcast_feat)).sortByKey().collect()

    [auc,mae,loss] = utils.get_eval_stat(eval_res,feat_weight,feat_dict,THETA)
    
    return [auc,mae,loss,eval_res]

def train(sc):

    global s
    global y

    feat_weight = {}

    learning_rate = 0.5
    ITER_MAX = 1000
    THETA = 4
    SAMPLING_RATE = 1

    [train_ins,train_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/train/part-00051")
    
    [eval_ins,eval_ins_count] = utils.load_ins(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_ins/eval/*")
    
    feat_dict = utils.load_feat(sc,"hdfs://hqz-ubuntu-master:9000/data/filtered_feat/*")
 
    cur_iter = 0
    
    [min_batch_ins,grad_vector,min_batch_count] = mini_batch_gradient(sc,train_ins,feat_weight,feat_dict,THETA,SAMPLING_RATE)
    
    d = grad_vector #* (-1)

    while cur_iter < ITER_MAX:
        
        print ( "iteration %d" % cur_iter )

        step = learning_rate / (math.sqrt(cur_iter+1))
        #step = line_search(min_batch_ins,sc,feat_weight,feat_dict,d,THETA)
        #step = line_search_backtracing(min_batch_ins,sc,feat_weight,feat_dict,d,THETA)

        si = d * step
        
        print ("step: %f" % step)

        update_weight(si,feat_dict,feat_weight)

        [auc, mae, loss,eval_res] = evaluate(sc,min_batch_ins,feat_weight,feat_dict,THETA)
        print ("selected %d samples: train_set : auc :%f, mae: %f, loss: %f" % (min_batch_count,auc,mae,loss))
        [auc, mae, loss,eval_res] = evaluate(sc,eval_ins,feat_weight,feat_dict,THETA)
        print ("test_set: auc :%f, mae: %f,loss: %f" % (auc,mae,loss))
        #utils.output(cur_iter, None, feat_weight,eval_res)
        
        [min_batch_ins,new_grad_vector,min_batch_count] = mini_batch_gradient(sc,train_ins,feat_weight,feat_dict,THETA,SAMPLING_RATE)
        
        if len(s) >= M:
            s.pop(0)

        s.append(si * (-1))

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

        cur_iter += 1

        grad_vector = new_grad_vector

        d =  get_Hk_gk(cur_iter,grad_vector)

if __name__ == "__main__":

    conf = SparkConf().setAppName("LR_SGD")
    sc = SparkContext(conf=conf)

    train(sc)
