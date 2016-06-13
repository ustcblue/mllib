import math
import traceback

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
    
        feat_num = len(self.feat)

        K = 0
        for i in range(0,feat_num):
            fi = self.feat[i]
            if fi in weights:
                weighted_sum += weights[fi][0]
                if K == 0 and len(weights[fi][1]) > 0:
                    K = len(weights[fi][1])

        for k in range(0,K):
            sum = 0
            sum_sqr = 0
            for i in range(0,feat_num):
                fi = self.feat[i]
                if fi in weights and len(weights[fi][1]) == K:
                    d = weights[fi][1][k]
                    sum += d
                    sum_sqr += d * d

            weighted_sum += 0.5 * (sum * sum - sum_sqr)

        print weighted_sum

        if weighted_sum >= 500:
            self.pred_val = 1.0
        elif weighted_sum <= -500:
            self.pred_val = 0.0
        else:
            self.pred_val = 1.0 / (1+math.exp(-weighted_sum))
        
        return self.pred_val
