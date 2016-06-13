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


