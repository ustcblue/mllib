import os
import sys
import numpy as np
import time
import random

class DataHelper:
    train_ins = []
    train_label = []
    
    eval_ins = []
    eval_label = []

    feat_dict = {}

    slots = []
    slot_size = 0

    vocabulary = {}
    vocab_size = 0

    ins_idx_list = []

    def __init__(self, _voc_size):
        random.seed(int(time.time()))
        self.vocab_size = _voc_size

    def read_ins(self, ins_file):
        label_array = []
        ins_array = []
    
        local_feat_dict = {}

        for line in open(ins_file,"r"):
        
            [lineid, show, clk, feat] = line.strip().split(" ",3)
        
            if int(clk) > 0:
                label_array.append([1])
            else:
                label_array.append([0])
        
            feat_str = feat.split(" ")
        
            ins = []

            for f in feat_str:
                f_segs = f.split(":")
                f_int = int(f_segs[0])
                slot_int = int(f_segs[1])

                ins.append(f_int)

                if f_int not in local_feat_dict:
                    local_feat_dict[f_int] = {"pv":0,"slot":slot_int}
            
                local_feat_dict[f_int]["pv"] += 1
            
            self.ins_idx_list.append(len(ins_array))
            ins_array.append(ins)

        return [label_array, ins_array, local_feat_dict]

    def gen_vocabulary(self):
    
        self.vocabulary = {}
        self.slots = []

        feat = []
        for f in self.feat_dict:
            feat.append([f, self.feat_dict[f]["pv"], self.feat_dict[f]["slot"]])
    
        feat = np.asarray(feat,dtype=np.uint64)

        sorted_feat = np.argsort(-feat,axis=0)

        feat_num = sorted_feat.shape[0]
        
        if self.vocab_size > feat_num :
            self.vocab_size = feat_num    

        for i in range(feat_num):
            
            if i >= self.vocab_size:
                break

            idx = sorted_feat[i][1]
            _slot = feat[idx][2]
            _feat = feat[idx][0]

            if _slot not in self.vocabulary:
                self.vocabulary[_slot] = { "feat_dict": {}, "idx": len(self.vocabulary) }
                self.slots.append(_slot)

            self.vocabulary[_slot]["feat_dict"][_feat] = len(self.vocabulary[_slot]["feat_dict"])
        
        self.slot_size = len(self.slots)

    def load_eval_ins(self, ins_file):
        [self.eval_label, self.eval_ins, tmp_feat_dict] = self.read_ins(ins_file)

    def load_train_ins_and_process(self, ins_file):

        [self.train_label, self.train_ins, self.feat_dict] = self.read_ins(ins_file)
        
        self.gen_vocabulary()

    def get_eval_ins_embedding(self, embedding_dict, embedding_size):

        embedding_ins = []

        for ins in self.eval_ins:
            _ins = np.zeros([self.slot_size, embedding_size], dtype=np.float32)
            
            for f in ins:
                if f in self.feat_dict:
                    slot = self.feat_dict[f]["slot"]
                    if slot in self.vocabulary:
                        slot_idx = self.vocabulary[slot]["idx"]
                        f_dict = self.vocabulary[slot]["feat_dict"]

                        if f in f_dict:
                            f_idx = f_dict[f]
                        else:
                            f_idx = len(f_dict)

                        _ins[slot_idx] += embedding_dict[slot_idx][f_idx]

            embedding_ins.append(_ins)
        
        return embedding_ins

    def get_eval_ins(self, batch_size, offset = 0):
        
        ins_array = []
        for s in self.slots:
            ins_array.append([])

        label_array = []

        idx_array = {}

        total_length = len(self.eval_ins)

        if offset + batch_size > total_length:
            batch_size = total_length - offset
            
        if batch_size > 0:
            idx = offset 
            while batch_size > 0:
                idx_array[idx] = 1
                idx += 1
                batch_size -= 1

        for idx in idx_array:
            _ins = self.eval_ins[ idx ]
            _label = self.eval_label[ idx ]

            ins = [ [] ] * self.slot_size

            for slot in self.vocabulary:
                idx = self.vocabulary[slot]["idx"]
                f_sz = len(self.vocabulary[slot]["feat_dict"])

                ins[idx] = np.zeros([f_sz+1], dtype = np.bool)

            for f in _ins:
                if f in self.feat_dict:
                    slot = self.feat_dict[f]["slot"]
                    if slot in self.vocabulary:
                        slot_idx = self.vocabulary[slot]["idx"]
                        f_sz = len(self.vocabulary[slot]["feat_dict"])
                        
                        if f in self.vocabulary[slot]["feat_dict"]:
                            f_idx = self.vocabulary[slot]["feat_dict"][f]
                        else:
                            f_idx = f_sz

                        ins[slot_idx][f_idx] = True

            for i in range(len(ins)):
                ins_array[i].append(ins[i])

            label_array.append(_label)


        return [ label_array, ins_array ]

    def get_next_batch(self, batch_size, offset = 0):

        ins_array = []
        for s in self.slots:
            ins_array.append([])

        label_array = []

        idx_array = {}

        total_length = len(self.train_ins)

        if offset + batch_size > total_length:
            batch_size = total_length - offset
            
        if batch_size > 0:
            idx = offset 
            while batch_size > 0:
                idx_array[idx] = 1
                idx += 1
                batch_size -= 1

        #print idx_array
        for idx in idx_array:

            _ins = self.train_ins[ self.ins_idx_list[idx] ]
            _label = self.train_label[ self.ins_idx_list[idx] ]

            ins = [ [] ] * self.slot_size

            for slot in self.vocabulary:
                idx = self.vocabulary[slot]["idx"]
                f_sz = len(self.vocabulary[slot]["feat_dict"])

                ins[idx] = np.zeros([f_sz+1], dtype = np.bool)

            for f in _ins:
                if f in self.feat_dict:
                    slot = self.feat_dict[f]["slot"]
                    if slot in self.vocabulary:
                        slot_idx = self.vocabulary[slot]["idx"]
                        f_sz = len(self.vocabulary[slot]["feat_dict"])
                        
                        if f in self.vocabulary[slot]["feat_dict"]:
                            f_idx = self.vocabulary[slot]["feat_dict"][f]
                        else:
                            f_idx = f_sz

                        ins[slot_idx][f_idx] = True

            for i in range(len(ins)):
                ins_array[i].append(ins[i])

            label_array.append(_label)

        return [ label_array, ins_array ]

    def shuffle_train_ins(self):
        ins_num = len(self.train_ins)
        dict = {}
        self.ins_idx_list = []

        while len(self.ins_idx_list) < ins_num:
            idx = random.randint(0, ins_num-1)
            if idx not in dict:
                dict[idx] = 1
                self.ins_idx_list.append(idx)

        
if __name__ == "__main__":
    
    d = DataHelper(1000000)
    d.shuffle_train_ins()
    print d.ins_idx_list

    #d.load_train_ins_and_process("data/train.50_51.ins")
    #d.get_next_batch(10,sequential = True)

    '''
    for s in d.vocabulary:
        sys.stdout.write("slot %d, idx %d:" % (s, d.vocabulary[s]["idx"]) )
        for f in d.vocabulary[s]["feat_dict"]:
            sys.stdout.write(" %d#%d" % (f,d.vocabulary[s]["feat_dict"][f]))
        sys.stdout.write("\n")
    '''
    '''
    offset = 0
    batch_size = 16

    print len(d.train_ins)

    while offset < len(d.train_ins):
        d.get_next_batch(batch_size, sequential = True, offset = offset)
        offset += batch_size
    '''
