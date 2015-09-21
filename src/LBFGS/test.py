import os
import sys

class testClass:
    var = 10.0

    def func_a(self,a,b):
        print self.var,a,b

    def func_b(self,c,fun):
        print c
        fun(1,2)

    def func_c(self):
        self.func_b(3,self.func_a)

class B:
    def printB(self,func):
        func()

ins=testClass()
b=B()

b.printB(ins.func_c)
#ins.func_c()
