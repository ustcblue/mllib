import os
import sys
import math

import scipy
import numpy as np
from scipy import sparse
    
class OWLQN:
    s = []
    y = []
    phro = []

    q = None
    
    diagH = None
    
    M = 0
    N = 0

    iter = 0
    bound = 0

    yy = 0
    ys = 0

    l1_weight = 0
    
    eval_func = None
    eval_func_for_test_set  = None

    x = None
    grad = None
    dir = None
    loss = 0
    
    tol = 0
    
    prev_Loss = []
    
    #@m : latest m updates
    #@n : dimension of the feature vector
    def __init__(self,m,n,l1_weight,eval_func, eval_func_for_test_set = None, tol = None):
        self.M = m
        self.N = n
        
        d = np.ones((n))
       
        self.diagH = sparse.diags(d,0)
        # to reference the element, use diag.data[0,0]

        self.iter = 0
        
        self.l1_weight = l1_weight
        self.eval_func = eval_func
        
        self.eval_func_for_test_set  = eval_func_for_test_set

        self.x = scipy.mat([0]*n).T
        self.grad = scipy.mat([0]*n).T
        self.dir = scipy.mat([0]*n).T
        self.loss = 0
        
        if tol == None:
            self.tol = 1e-4
        else:
            self.tol = tol
    def push_limit_array(self,array,item,limit):
        
        array.append(item)
        if len(array) > limit:
            array.pop(0)
    
    def lb1 ( self, term_msg , eval_mae, eval_auc):
        print "Iter: %d : Loss %f\t Term_Criterion: %s\t Eval_MAE: %f \t Eval_AUC: %f" % (self.iter, self.loss, term_msg, eval_mae, eval_auc )
        
    def makeSteepestDescDir(self):
        if self.l1_weight == 0:
            self.dir = - self.grad.copy()
        else:
            self.dir = self.grad.copy()

            for i in range(0,self.N,1):
                if self.x[i,0] < 0:
                    self.dir[i,0] = -self. grad[i,0] + self.l1_weight
                elif self.x[i,0] > 0:
                    self.dir[i,0] = - self.grad[i,0] - self.l1_weight
                else:
                    if self.grad[i,0] < - self.l1_weight:
                        self.dir[i,0] = - self.grad[i,0] - self.l1_weight
                    elif self.grad[i,0] > self.l1_weight:
                        self.dir[i,0] = - self.grad[i,0] + self.l1_weight
                    else:
                        self.dir[i,0] = 0

    def eval_L1(self, input_x):
        [ loss, g_mat ] = self.eval_func(input_x)

        if self.l1_weight > 0:
            for i in range(1,self.N,1):
                loss += math.fabs(input_x[i-1,0]) * self.l1_weight

        return [loss, g_mat]
    
    def get_next_point(self, alpha, newX):
        newX[0] = self.x + self.dir* alpha
        if self.l1_weight > 0:
            for i in range(0,self.N):
                if newX[0][i,0] * self.x[i,0] < 0.0:
                    newX[0][i,0] = 0.0
    
    def DirDeriv(self):
        if self.l1_weight == 0:
            return self.dir.T * self.grad
        else:
            value = 0
            for i in range(0,self.N):
                if self.dir[i,0] != 0:
                    if self.x[i,0] < 0:
                        value += self.dir[i,0] * ( self.grad[i,0] - self.l1_weight )
                    elif self.x[i,0] > 0:
                        value += self.dir[i,0] * ( self.grad[i,0] + self.l1_weight )
                    elif self.dir[i,0] < 0:
                        value += self.dir[i,0] * ( self.grad[i,0] - self.l1_weight )
                    elif self.dir[i,0] > 0:
                        value += self.dir[i,0] * ( self.grad[i,0] + self.l1_weight )
            return value
        
    def line_search(self, newX, newGradient):
        orgDirDeriv = self.DirDeriv()
        if orgDirDeriv >= 0:
            print "LBFGS chose a non-descent direction: check your gradient !"
            return -1
        
        alpha = 1.0
        backoff = 0.5
        
        if self.iter == 1:
            normDir = math.sqrt((self.dir.T * self.dir)[0])
            alpha = 1.0 / normDir
            backoff = 0.1
        
        c1 = 1e-4
        
        while True:
            self.get_next_point(alpha, newX)
            [newLoss, newGradient[0]] = self.eval_L1(newX[0])
            
            if newLoss <= self.loss + c1 * orgDirDeriv * alpha:
                break
            
            alpha *= backoff
        
        self.loss = newLoss
    
    def termCrit(self):
        
        ret = 1e10
        
        if len(self.prev_Loss) > 5:
            firstLoss = self.prev_Loss[0]
            averageImprovement = (firstLoss - self.loss) / len(self.prev_Loss)
            ret = averageImprovement / math.fabs(self.loss)
            msg = " (%f) " % ret
        else:
            msg = " (waiting for five iters) "
        
        self.push_limit_array(self.prev_Loss, self.loss, 10)
        return [ ret, msg ]
    
    def owlqn(self, init_x, eval_func_for_test_set = None):

        self.x = scipy.mat(init_x).T
        
        [ self.loss , self.grad ]= self.eval_L1(self.x)
        
        [ _term_criterion, msg ] = self.termCrit()
        
        eval_mae = 0
        eval_auc = 0

        if self.eval_func_for_test_set != None:
            [eval_mae, eval_auc] = self.eval_func_for_test_set(self.x)
        
        self.lb1(msg,eval_mae,eval_auc)
        
        self.iter = self.iter + 1
        self.bound = self.iter - 1

        newX = [ scipy.mat([0]*self.N) ]
        newGrad = [ scipy.mat([0]*self.N) ]
        
        while True:

            self.makeSteepestDescDir()
                
            if self.iter != 1:
                
                self.push_limit_array(self.phro,1.0/self.ys,self.M)
                
                q = self.dir

                alpha = [ 0 ] * self.bound

                for i in range(self.bound-1,-1,-1):
                    alpha[i] = (self.s[i].T * self.dir * self.phro[i])[0,0]
                    self.dir = self.dir - alpha[i] * self.y[i]

                self.dir = self.diagH * self.dir

                for i in range(0,self.bound):
                    beta = self.phro[i] * self.y[i].T * self.dir
                    self.dir = self.dir + self.s[i] * (alpha[i] - beta)

                #FixDirSign
                if self.l1_weight > 0:
                    for i in range(0,self.N,1):
                        if self.dir[i,0] * q[i,0] <= 0:
                            self.dir[i,0] = 0

            w = self.grad
            
            self.line_search(newX, newGrad)
            
            if self.eval_func_for_test_set != None:
                [eval_mae, eval_auc] = self.eval_func_for_test_set(self.x)

            self.push_limit_array(self.s, newX[0] - self.x, self.M)
            self.push_limit_array(self.y,newGrad[0] - self.grad,self.M)

            self.x = newX[0]
            self.grad = newGrad[0]
            
            [ _term_criterion, msg ] = self.termCrit()
            self.lb1(msg, eval_mae, eval_auc)

            if _term_criterion < self.tol:
                print "optimization done..."
                break
                
            self.iter += 1
            self.bound = self.iter - 1

            if self.iter > self.M:
                self.bound = self.M
            
            self.ys = (self.y[-1].T * self.s[-1])[0,0]
            
            self.yy = (self.y[-1].T * self.y[-1])[0,0]

            d = np.ones((self.N)) * (self.ys / self.yy)
            self.diagH = sparse.diags(d,0)
            
if __name__=="__main__":
    
    n=100
    m=5

    owlqn_instance = OWLQN(m,n,4,test_func)

    x = [0] * n

    for j in range(1,n+1,2):
        x[j - 1] = - 1.2
        x[j + 1 - 1] = 1.0

    owlqn_instance.owlqn(x)
