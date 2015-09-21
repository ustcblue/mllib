import os
import sys
import math
#from sympy.concrete.tests.test_delta import dp

class Mcsrch:
    
    infoc = [0]
    
    dg = 0.0
    dgm = 0.0
    dginit = 0.0
    dgtest = 0.0
    dgx = [0.0]
    dgxm = [0.0]
    dgy = [0.0]
    dgym = [0.0]
    finit = 0.0
    ftest1 = 0.0
    fm = 0.0
    fx = [0.0]
    fxm = [0.0]
    fy = [0.0]
    fym = [0.0]
    p5 = 0.0
    p66 = 0.0
    stx = [0.0]
    sty = [0.0]
    stmin = 0.0
    stmax = 0.0
    width = 0.0
    width1 = 0.0
    xtrapf = 0.0
    brackt = [False]
    stage1 = False
    
    def sqr(self,  x ):
        return x*x
    
    def min(self,x,y):
        if x < y:
            return x
        else:
            return y
    
    def max(self,x,y):
        if x < y:
            return y
        else:
            return x
        
    def max3(self, x, y, z ):
        if x < y:
            if y < z:
                return z
            else:
                return y
        else:
            if x < z:
                return z
            else:
                return x
  
    def mcsrch(self , x , f , g , si , stp , func , ftol, gtol, xtol, maxfev, stpmin, stpmax,nfev,info,ms_print):
        self.p5 = 0.5
        self.p66 = 0.66
        self.xtrapf = 4
        
        self.infoc = [1]
        
        if len(x[0]) == 0 or len(g[0]) == 0 or len(si) == 0 or stp <= 0 or gtol < 0 or xtol < 0 or stpmin < 0 or stpmax < stpmin or maxfev <= 0:
            return
        
        self.dginit = 0
        
        self.dginit = (g[0].T * si)[0,0]
        
        if self.dginit >= 0:
            print "The search direction is not a descent direction"
            str = raw_input("press any key to continue: ");
            return
        
        self.brackt[0] = False
        self.stage1 = True
        
        nfev[0] = 0
        self.finit = f[0]
        self.dgtest = ftol * self.dginit
        
        self.width = stpmax - stpmin
        self.width1 = self.width / self.p5
        
        wa = x[0]
        
        self.stx[0] = 0.0
        self.fx[0] = self.finit
        self.dgx[0] = self.dginit
        self.sty[0] = 0.0
        self.fy[0] = self.finit
        self.dgy[0] = self.dginit
        
        while True:
            if self.brackt[0]:
                self.stmin = self.min(self.stx[0], self.sty[0])
                self.stmax = self.max(self.stx[0],self.sty[0])
            else:
                self.stmin = self.stx[0]
                self.stmax = stp[0] + self.xtrapf * (stp[0] - self.stx[0])
            
            stp[0] = self.max(stp[0],stpmin)
            stp[0] = self.min(stp[0],stpmax)
            
            if (self.brackt[0] and (stp[0] <= self.stmin or stp[0] >= self.stmax)) or nfev[0] >= maxfev - 1 or self.infoc[0] == 0 or (self.brackt[0] and self.stmax - self.stmin <= xtol * self.stmax):
                stp[0] = self.stx[0]
            
            x[0] = wa + si * stp[0]
            
            [f[0],g[0]] = func(x[0])

            if ms_print:
                print "\t\t\t\t\t\t\t\t\t\t\t\t\tms\t%d\t%f\t%f\t%f" % (nfev[0],f[0],math.sqrt((g[0].T*g[0])[0,0]),stp[0])

            nfev[0] += 1
            
            self.dg = (g[0].T * si)[0,0]
            
            self.ftest1 = self.finit + stp[0] * self.dgtest
            
            if (self.brackt[0] and (stp[0] <= self.stmin or stp[0] >= self.stmax)) or self.infoc[0] == 0:
                '''
                Rounding errors prevent further progress.
                There may not be a step which satisfies the
                sufficient decrease and curvature conditions.
                Tolerances may be too small.
                '''
                info[0] = 6
            if stp[0] == stpmax and f[0] <= self.ftest1 and self.dg <= self.dgtest:
                '''
                The step is at the upper bound <code>stpmax</code>.
                '''
                info[0] = 5
            if stp[0] == stpmin and (f[0] >= self.ftest1 or self.dg >= self.dgtest):
                '''
                The step is at the lower bound <code>stpmin</code>.
                '''
                info[0] = 4
            if nfev[0] >= maxfev:
                '''
                Number of function evaluations has reached <code>maxfev</code>.
                '''
                info[0] =3
            if self.brackt[0] and self.stmax - self.stmin <= xtol * self.stmax:
                '''
                Relative width of the interval of uncertainty is at most xtol
                '''
                info[0] =2
            if f[0] <= self.ftest1 and math.fabs(self.dg) <= gtol * (-self.dginit):
                '''
                The sufficient decrease condition and the directional derivative condition hold.
                '''
                info[0] = 1
                
            if info[0] != 0:
                return
            
            if self.stage1 and f[0] <= self.ftest1 and self.dg >= self.min(ftol,gtol) * self.dginit:
                self.stage1 = False
            
            if self.stage1 and f[0] <= self.fx[0] and f[0] > self.ftest1:
                self.fm = f[0] - stp[0] * self.dgtest
                self.fxm[0] = self.fx[0] - self.stx[0] * self.dgtest
                self.fym[0] = self.fy[0] - self.sty[0] * self.dgtest
                self.dgm = self.dg - self.dgtest
                self.dgxm[0] = self.dgx[0] - self.dgtest
                self.dgym[0] = self.dgy[0] - self.dgtest
                
                self.mcstep(self.stx,self.fxm,self.dgxm,self.sty,self.fym,self.dgym,stp,self.fm,self.dgm,self.brackt,self.stmin,self.stmax,self.infoc)
                
                self.fx[0] = self.fxm[0] + self.stx[0] * self.dgtest
                self.fy[0] = self.fym[0] + self.sty[0] * self.dgtest
                self.dgx[0] = self.dgxm[0] + self.dgtest
                self.dgy[0] = self.dgym[0] + self.dgtest
            else:
                self.mcstep(self.stx,self.fx,self.dgx,self.sty,self.fy,self.dgy,stp,f[0],self.dg,self.brackt,self.stmin,self.stmax,self.infoc)
            
            if self.brackt[0]:
                if math.fabs(self.sty[0] - self.stx[0]) >= self.p66 * self.width1:
                    stp[0] = self.stx[0] + self.p5 * (self.sty[0] - self.stx[0])
                self.width1 = self.width
                self.width = math.fabs(self.sty[0] - self.stx[0])
                
    def mcstep(self,stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax,info):
        
        info[0] = 0
        
        if ( brackt[0] and ( stp[0] <= self.min(stx[0], sty[0]) or stp[0] >= self.max(stx[0], sty[0])) ) or dx[0] * (stp[0] - stx[0]) >= 0.0 or stpmax < stpmin:
            return
        
        sgnd = dp * (dx[0] / math.fabs(dx[0]))
        
        if fp > fx[0]:
            '''
            First case. A higher function value.
            The minimum is bracketed. If the cubic step is closer
            to stx than the quadratic step, the cubic step is taken,
            else the average of the cubic and quadratic steps is taken.
            '''
            info[0] = 1
            bound = True
            theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp
            s = self.max3(math.fabs(theta),math.fabs(dx[0]),math.fabs(dp))
            gamma = s * math.sqrt(self.sqr(theta/s)-(dx[0]/s) *(dp/s))
            
            if stp[0] < stx[0]:
                gamma = -gamma
            
            p = (gamma - dx[0]) + theta
            q = ( (gamma - dx[0]) + gamma) + dp
            r = p / q
            
            stpc = stx[0] + r * (stp[0] - stx[0])
            stpq = stx[0] + ( ( dx[0] / ( ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] ) ) / 2 ) * ( stp[0] - stx[0] )
            
            if math.fabs(stpc - stx[0]) < math.fabs(stpq - stx[0]):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2
            
            brackt[0] = True
        elif sgnd < 0.0:
            '''
            Second case. A lower function value and derivatives of
            opposite sign. The minimum is bracketed. If the cubic
            step is closer to stx than the quadratic (secant) step,
            the cubic step is taken, else the quadratic step is taken.
            '''
            info[0] = 2
            bound = False
            theta = 3 * ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] + dp
            s = self.max3 ( math.fabs ( theta ) , math.fabs ( dx[0] ) , math.fabs ( dp ) )
            gamma = s * math.sqrt ( self.sqr( theta / s ) - ( dx[0] / s ) * ( dp / s ) )
            if  stp[0] > stx[0]:
                gamma = - gamma
            p = ( gamma - dp ) + theta
            q = ( ( gamma - dp ) + gamma ) + dx[0]
            r = p/q
            stpc = stp[0] + r * ( stx[0] - stp[0] )
            stpq = stp[0] + ( dp / ( dp - dx[0] ) ) * ( stx[0] - stp[0] )
            if  math.fabs ( stpc - stp[0] ) > math.fabs ( stpq - stp[0] ) :
                stpf = stpc
            else:
                stpf = stpq
            brackt[0] = True
        elif math.fabs ( dp ) < math.fabs ( dx[0] ) :
            '''
            Third case. A lower function value, derivatives of the
            same sign, and the magnitude of the derivative decreases.
            The cubic step is only used if the cubic tends to infinity
            in the direction of the step or if the minimum of the cubic
            is beyond stp. Otherwise the cubic step is defined to be
            either stpmin or stpmax. The quadratic (secant) step is also
            computed and if the minimum is bracketed then the the step
            closest to stx is taken, else the step farthest away is taken.
            '''
            info[0] = 3
            bound = True
            theta = 3 * ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] + dp
            s = self.max3 ( math.fabs ( theta ) , math.fabs ( dx[0] ) , math.fabs ( dp ) )
            gamma = s * math.sqrt ( self.max ( 0, self.sqr( theta / s ) - ( dx[0] / s ) * ( dp / s ) ) )
            if stp[0] > stx[0] :
                gamma = - gamma
            p = ( gamma - dp ) + theta
            q = ( gamma + ( dx[0] - dp ) ) + gamma
            r = p/q
            if r < 0.0 and gamma != 0.0:
                stpc = stp[0] + r * ( stx[0] - stp[0] )
            elif stp[0] > stx[0]:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp[0] + ( dp / ( dp - dx[0] ) ) * ( stx[0] - stp[0] )
            if  brackt[0] :
                if math.fabs ( stp[0] - stpc ) < math.fabs ( stp[0] - stpq ):
                    stpf = stpc
                else:
                    stpf = stpq
            else:
                if math.fabs ( stp[0] - stpc ) > math.fabs ( stp[0] - stpq ) :
                    stpf = stpc
                else:
                    stpf = stpq
        else:
            ''' 
            Fourth case. A lower function value, derivatives of the
            same sign, and the magnitude of the derivative does
            not decrease. If the minimum is not bracketed, the step
            is either stpmin or stpmax, else the cubic step is taken.
            '''
            info[0] = 4
            bound = False
            if brackt[0]:
                theta = 3 * ( fp - fy[0] ) / ( sty[0] - stp[0] ) + dy[0] + dp
                s = self.max3 ( math.fabs ( theta ) , math.fabs ( dy[0] ) , math.fabs ( dp ) )
                gamma = s * math.sqrt ( self.sqr( theta / s ) - ( dy[0] / s ) * ( dp / s ) )
                if stp[0] > sty[0]:
                    gamma = - gamma
                p = ( gamma - dp ) + theta
                q = ( ( gamma - dp ) + gamma ) + dy[0]
                r = p/q
                stpc = stp[0] + r * ( sty[0] - stp[0] )
                stpf = stpc
            elif stp[0] > stx[0]:
                stpf = stpmax
            else:
                stpf = stpmin
        
        if fp > fx[0]:
            sty[0] = stp[0]
            fy[0] = fp
            dy[0] = dp
        else:
            if sgnd < 0.0:
                sty[0] = stx[0]
                fy[0] = fx[0]
                dy[0] = dx[0]
            stx[0] = stp[0]
            fx[0] = fp
            dx[0] = dp
            
        '''Compute the new step and safeguard it.'''
        
        stpf = self.min ( stpmax , stpf )
        stpf = self.max ( stpmin , stpf )
        stp[0] = stpf

        if brackt[0] and bound:
            if sty[0] > stx[0]:
                stp[0] = self.min ( stx[0] + 0.66 * ( sty[0] - stx[0] ) , stp[0] )
            else:
                stp[0] = self.max ( stx[0] + 0.66 * ( sty[0] - stx[0] ) , stp[0] )
        return
