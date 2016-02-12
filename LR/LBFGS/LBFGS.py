import os
import sys
import math

import scipy
import numpy as np
from scipy import sparse

sys.path.append("../utils/")
from Mcsrch import Mcsrch
'''

'''

def test_func(x):
    f = 0
    shape = x.shape
    g = [0] * shape[0]
    for j in range(1,shape[0]+1,2):
        t1 = 1.e0 - x [j -1,0]
        t2 = 1.e1 * ( x [ j + 1 -1,0] - x [ j -1,0] * x[ j-1,0] )
        g [ j + 1 -1] = 2.e1 * t2
        g [ j -1] = - 2.e0 * ( x [j -1,0] * g [ j + 1 -1] + t1 )
        f= f+t1*t1+t2*t2
        
    g_mat = scipy.mat(g).T
    
    return [ f, g_mat ]
    
class LBFGS:
    s = []
    y = []
    phro = []

    q = None
    
    diagH = None
    
    M = 0
    N = 0

    iter = 0
    bound = 0

    maxfev = 0
    ftol = 0
    stp1 = 0
    gnorm = 0
    nfun = 0

    yy = 0
    ys = 0

    gtol = 0
    xtol = 0
    maxfev = 0
    stpmin = 0
    stpmax = 0
    finish = False
    eps = 0
    
    info = [0]
    stp = [0.0]
    nfev = [0]
    iprint = [1,0]
    mcsrch_instance = None

    #@m : latest m updates
    #@n : dimension of the feature vector
    def __init__(self,m,n):
        self.M = m
        self.N = n
        
        d = np.ones((n))

        self.diagH = sparse.diags(d,0)
        # to reference the element, use diag.data[0,0]

        self.iter = 0
        self.maxfev = 20
        self.ftol = 0.0001
        self.gtol = 0.9
        self.nfun = 1
        self.finish = False
        self.eps = 1.0e-5
        self.iprint = [1,0]
        
        self.stpmax = 1e20
        self.stpmin = 1e-20
        self.xtol = 1e-16
        
        self.mcsrch_instance = Mcsrch()

    def push_limit_array(self,array,item,limit):
        
        array.append(item)
        if len(array) > limit:
            array.pop(0)
    
    def lb1 ( self, iprint , nfun , gnorm , x , f , g , eval_mae, eval_auc, stp , finish ):

        if self.iter == 0:
            print( "*************************************************" )
            print( " n = %d  number of corrections = %d\n   initial values f=%f gnorm=%f"  % (self.N,self.M,f,gnorm))
            if  iprint [ 2 -1] >= 1:
                str = " vector x =" 
                for  i in range(1,self.N,1):
                    str += ( "  %f" % x[i-1] )
                print str
                
                str = " gradient vector g =" 
                for i in range(1,self.N,1):
                    str += (" %f" % g[i-1] )
                print str
            print( "*************************************************" )
            print( "\ti\tnfn\tfunc\tgnorm\tsteplength\teval_mae\teval_auc" )
        else:
            if ( iprint [ 1 -1] == 0 ) and ( self.iter != 1 and finish != True ):
                return
            if iprint [ 1 -1] != 0:
                if  (self.iter - 1) % iprint [ 1 -1] == 0 or finish == True:
                    if  iprint [ 2 -1] > 1 and self.iter > 1 :
                        print( "\ti\tnfn\tfunc\tgnorm\tsteplength\teval_mae\teval_auc" );
                    print( "\t%d\t%d\t%f\t%f\t%f\t%f\t%f" % (self.iter,nfun,f,gnorm,stp[0],eval_mae, eval_auc ) )
                else:
                    return
            else:
                if iprint [ 2 -1] > 1 and finish == True: 
                        print( "\ti\tnfn\tfunc\tgnorm\tsteplength\teval_mae\teval_auc" )
                print( "\t%d\t%d\t%f\t%f\t%f\t%f\t%f" % (self.iter,nfun,f,gnorm,stp[0], eval_mae, eval_auc) )

            if  iprint [ 2 -1] == 2 or iprint [ 2 -1] == 3:
                if finish:
                    str = " final point x =" 
                else:
                    str = " vector x =  " 
                
                for i in range(1,self.N,1):
                    str += ( "  %f" % x[i-1] )
                print str
                if  iprint [ 2 -1] == 3 :
                    str = ( " gradient vector g =" )
                    for i in range(1,self.N,1):
                        str += " %f" % g[i-1]
                    print str
            if finish:
                print( " The minimization terminated without detecting errors. iflag = 0" );
        return
    
    def lbfgs(self, init_x, eval_func_for_train_set, eval_func_for_test_set):
        
        f = [0]
        g = [None]
        x = [scipy.mat(init_x).T]
        
        [f[0], g[0]]= eval_func_for_train_set(x[0])

        [eval_mae, eval_auc] = eval_func_for_test_set(x[0])

        if self.iter == 0:
            si = - self.diagH * g[0]
            self.s.append(si)
            self.gnorm = math.sqrt( g[0].T * g[0] )
            self.stp1 = 1/self.gnorm

            #print self.gnorm
            #print f[0]
            
            self.lb1(self.iprint, self.nfun, self.gnorm, x[0], f[0], g[0], eval_mae, eval_auc, self.stp, self.finish)
            
            self.iter = self.iter + 1
            self.bound = self.iter - 1

        while True:

            if self.iter != 1:
                
                self.push_limit_array(self.phro,1.0/self.ys,self.M)
                
                q = - g[0]

                alpha = [ 0 ] * self.bound

                for i in range(self.bound-1,-1,-1):
                    alpha[i] = (self.s[i].T * q * self.phro[i])[0,0]
                    q = q - alpha[i] * self.y[i]

                q = self.diagH * q

                for i in range(0,self.bound):
                    beta = self.phro[i] * self.y[i].T * q
                    q = q + self.s[i] * (alpha[i] - beta)

                self.push_limit_array(self.s,q,self.M)
                
            if self.iter == 1:
                self.stp[0] = self.stp1
            else:
                self.stp[0] = 1

            w = g[0]

            self.nfev[0] = 0
            
            self.mcsrch_instance.mcsrch(x, f, g, self.s[-1], self.stp, eval_func_for_train_set, self.ftol, self.gtol, self.xtol, self.maxfev, self.stpmin, self.stpmax, self.nfev, self.info,False)

            [eval_mae, eval_auc] = eval_func_for_test_set(x[0])

            if self.info[0] == -1:
                print self.iter,self.info[0]
                return 1

            if self.info[0]!= 1:
                print self.iter,self.info[0]
                return -1

            self.nfun += self.nfev[0]

            self.s[-1] = self.stp[0] * self.s[-1]
            
            self.push_limit_array(self.y,g[0]-w,self.M)

            self.gnorm = math.sqrt(g[0].T * g[0])
            self.xnorm = math.sqrt(x[0].T * x[0])
            if self.xnorm <= 1.0:
                self.xnorm = 1.0
            
            if self.gnorm / self.xnorm <= self.eps:
                self.finish = True
            
            self.lb1(self.iprint, self.nfun, self.gnorm, x[0], f[0], g[0], eval_mae, eval_auc, self.stp, self.finish)
            
            if self.finish:
                return
            
            self.iter += 1
            self.info[0] = 0
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

    lbfgs_instance = LBFGS(m,n)

    x = [0] * n

    for j in range(1,n+1,2):
        x[j - 1] = - 1.2
        x[j + 1 - 1] = 1.0

    lbfgs_instance.lbfgs(x,test_func)
