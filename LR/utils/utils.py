import math
from pyspark import SparkContext, SparkConf
from pyspark import AccumulatorParam

class WeightAccumulatorParam(AccumulatorParam):
    def zero(self,initialValue):
        return {}

    def addInPlace(self,v1,v2):
        for k in v2:
            if k not in v1:
                v1[k] = v2[k]
            else:
                v1[k] += v2[k]

        return v1

class Instance:
    label = None
    feat = None
    pred_val = None

    def __init__(self,line):

        segs = line.strip().split(" ")

        self.label = int(segs[2])
        
        self.feat = []
        
        self.pred_val = -1

        for s in segs:
            s_s = s.split(":")
            if len(s_s) == 2:
                self.feat.append(eval(s_s[0]))

    def predict(self,weights,cache = 0):

        if cache == 1 and self.pred_val > -1:
            return self.pred_val

        weighted_sum = 0
    
        for f in self.feat:
            if f in weights:
                weighted_sum += weights[f]

        if weighted_sum >= 30:
            self.pred_val = 1.0
        elif weighted_sum <= -30:
            self.pred_val = 0.0
        else:
            self.pred_val = 1.0 / (1+math.exp(-weighted_sum))

        return self.pred_val


def load_ins(sc,url):
    instanceFile=sc.textFile(url)
    ins = instanceFile.map(lambda s: Instance(s)).cache()
    ins_count = ins.count()
    
    return [ins,ins_count]

def load_feat(sc,url):
    featFile=sc.textFile(url)
    feat = featFile.map(lambda s: int(s)).collect()

    feat_count = len(feat)
    feat_dict = {}

    for i in range(0,feat_count):
        feat_dict[feat[i]] = i

    return feat_dict

def evalulate_map(ins,broadcast_feat):

    feat_weight = broadcast_feat.value

    return (ins.predict(feat_weight),ins.label)

def get_eval_stat(sorted_res):
    auc = 0
    mae = 0

    #sorted_res = sorted(eval_res,key=lambda p: p[1])
    
    idx = 0

    rank = 0
    M = 0
    N = 0
    
    count = len(sorted_res)
    
    loss = 0

    while idx < count:
        mae += math.fabs(sorted_res[idx][1] - sorted_res[idx][0])
        
        if sorted_res[idx][1] == 0:
            loss += - math.log(1-sorted_res[idx][0])

            N = N + 1
        elif sorted_res[idx][1] == 1:
            loss += - math.log(sorted_res[idx][0])
            M = M + 1
            rank = rank + idx + 1
            
        idx += 1

    mae = mae / count

    auc = (rank*1.0 - M*(M+1)/2.0) / ( M * N * 1.0 )
    
    return [ auc, mae,loss ] 

def output(iter_idx, grad, feat_weight,eval_res):

    if grad != None:
        grad_file_name = "weight/grad_%d" % iter_idx
    
        grad_file = open(grad_file_name,"w")

        for g in grad:
            grad_file.write("%d\t%f\n"%(g,grad[g]))

        grad_file.close()
    
    weight_file_name = "weight/weight_%d" % iter_idx
    weight_file = open(weight_file_name,"w")
    
    #feat_weight = feat_weight_dict.value
    
    for f in feat_weight:
        weight_file.write("%d\t%f\n"%(f,feat_weight[f]))

    weight_file.close()

    eval_file_name =  "weight/res_eval_%d" % iter_idx
    eval_file = open(eval_file_name,"w")

    for e in eval_res:
        eval_file.write("%f\t%d\n" % (e[0],e[1]))
    eval_file.close()
