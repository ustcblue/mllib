import types
import copy

class Matrix(list):
    
    row = 0
    col = 0

    def __init__(self,value=None,row=0,col=0,idx_dict=None):
 
        if isinstance(value,Matrix):
            list.__init__([])
            self.row = value.row
            self.col = value.col
            self.extend(copy.deepcopy(value))

        elif isinstance(value,list):
            if len(value) > 0:
                if isinstance(value[0],list):
                    if len(value[0]) > 0:
                        list.__init__([])
                        self.row = len(value)
                        self.col = len(value[0])
                        self.extend(copy.deepcopy(value))
                        #self.extend(value)
                        return
            raise TypeError("error matrix dim")
        
        elif isinstance(value,dict):

            if isinstance(idx_dict,dict):
                self.col = 1
                self.row = len(idx_dict)
                for i in range(0,self.row):
                    self.append( [0] )
                
                #print self

                for k in value:
                    if k not in idx_dict:
                        raise TypeError("error index")
                
                    self[idx_dict[k]][0] = value[k]
            else:
                raise TypeError("idx_dict needed")

        elif isinstance(value,int) or isinstance(value,float):
            if row == 0 or col == 0:
                raise TypeError("error matrix dim")

            self.row = row
            self.col = col

            list.__init__([])

            for i in range(0,row):
                self.append( [value]*self.col )

    def __add__(self,v):
        if isinstance(v,Matrix):
            if self.row != v.row or self.col != v.col:
                raise TypeError("matrix dims don't match")
            
            ret = copy.deepcopy(self)
            
            for i in range(0,self.row):
                for j in range(0,self.col):
                    ret[i][j] += v[i][j]
            
            return ret

    def __iaddr__(self,v):
            if self.row != v.row or self.col != v.col:
                raise TypeError("matrix dims don't match")
            
            for i in range(0,self.row):
                for j in range(0,self.col):
                    self[i][j] += v[i][j]
            
            return self

    def __sub__(self,v):
        if isinstance(v,Matrix):
            if self.row != v.row or self.col != v.col:
                raise TypeError("matrix dims don't match")
            
            ret = copy.deepcopy(self)
            
            for i in range(0,self.row):
                for j in range(0,self.col):
                    ret[i][j] -= v[i][j]
            
            return ret

    def __isub__(self,v):
         if isinstance(v,Matrix):
            if self.row != v.row or self.col != v.col:
                raise TypeError("matrix dims don't match")
            
            for i in range(0,self.row):
                for j in range(0,self.col):
                    self[i][j] -= v[i][j]
            
            return self

    def __mul__(self,v):
        
        if isinstance(v,int) or isinstance(v,float):    
            ret = copy.deepcopy(self)
            
            for i in range(0,ret.row):
                for j in range(0,ret.col):
                    ret[i][j] *= v
            
            return ret

        elif isinstance(v,Matrix):

            ret = Matrix(0,self.row,v.col)

            if self.col != v.row:
                raise TypeError("matrix dims don't match")
                return None

            for i in range(0,self.row):
                for j in range(0,v.col):
                    for k in range(0,self.col):
                        ret[i][j] += self[i][k] * v[k][j]

            return ret
    
    def Trans(self):

        ret = Matrix(0,self.col,self.row)

        for i in range(0,self.row):
            for j in range(0,self.col):
                ret[j][i] = self[i][j]

        return ret

if __name__ == '__main__':
    a=[[1,2,3],[2,3,4]]
    b=Matrix(a)
    c=Matrix(b)

    d=Matrix(value={"asdf":1.0,"qwer":2.0},idx_dict={"zxcv":0,"asdf":1,"sgwert":2,"qwer":3})

    b[0][1]=5
    print b
    print c

    print d
