# -*- coding: utf-8 -*-

##
##    huajiao.py: This is a toy tool for integer-valued autoregressive model.
##
##    Copyright (C) 2017 Tamio KOYAMA
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

import numpy as np

class INAR:
    """ Integer-valued autoregressive model without inflow"""

    def __init__(self, m):
        if isinstance(m, int):
            self.m     = m
        else:
            self.m = 0
        self.r     = np.zeros(self.m)
        self._state = np.zeros(self.m)
        return

    def getSampleData(self, N):
        return self._getSampleData(N)

    def estimateParameters(self, x):
        self.r = self._conditionalLeastSquareEstimation(x)
        return 

    def _newX(self):
        x = 0
        #print self._state
        for i in range(self.m):
            rx = np.random.binomial(n=self._state[i], p=self.r[i], size=1)
            #print ["in _newX", i, self._state[i], self.r[i], rx]
            x += rx[0]
        return x
    
    def _updateState(self):
        x = self._newX()
        for i in range(self.m-1):
            self._state[self.m-i-1] = self._state[self.m-i-2]
        self._state[0] = x
        #print self._state
        return self._state[0]

    def _getSampleData(self, N):
        retv = np.zeros(N)
        for i in range(N):
            retv[i] = self._updateState()
        return retv

    def _setParameterGamma(self, x):
        N = len(x)
        p = self.m
        gamma = np.zeros((p+1)*(p+1)).reshape(p+1,p+1)
        for i in range(p+1):
            for j in range(i,p+1):
                #print ["foo",i+1,j+1]
                #print x[(p-i):(N-i)]
                #print x[(p-j):(N-j)]
                gamma[i][j] = np.mean(x[(p-i):(N-i)]*x[(p-j):(N-j)])
                gamma[j][i] = gamma[i][j]
        return gamma

    def _getNormalEquation(self, x):
        N = len(x)
        p = self.m
        gamma = self._setParameterGamma(x)
        #print ["gamma", gamma]
        v = np.zeros(p)
        mat = np.zeros(p*p).reshape(p,p)

        for i in range(p):
            v[i] = gamma[i+1][0]
        for i in range(p):
            for j in range(p):
                mat[i][j] = gamma[i+1][j+1]
        return [mat,v]
    
    def _conditionalLeastSquareEstimation(self, x):
        if len(x) == 0:
            return 0
        retv = self._getNormalEquation(x)
        #print retv[0]
        #print retv[1]
        imat = np.linalg.inv(retv[0])
        w = imat.dot(retv[1])
        return w

class INARinflow(INAR):
    """ Integer-valued autoregressive model"""
    def __init__(self, p, Inflow):
        if isinstance(p, int):
            self.m     = p
        else:
            self.m = 0
        self.r     = np.zeros(self.m)
        self._state = np.zeros(self.m)
        if isinstance(Inflow, inflow):
            self.inflow = Inflow
        else:
            print "Error"
        return

    def _newX(self):
        x = self.inflow.getSample()
        #print self._state
        for i in range(self.m):
            rx = np.random.binomial(n=self._state[i], p=self.r[i], size=1)
            #print ["in _newX", i, self._state[i], self.r[i], rx]
            x += rx[0]
        return x

    def _setParameterBarX(self, x):
        N = len(x)
        p = self.m
        barx = np.zeros(p+1)
        for i in range(p+1):
            #print x[(p-i):(N-i)]
            barx[i] = np.mean(x[(p-i):(N-i)])
        return barx

    def _getNormalEquation(self, x):
        N = len(x)
        p = self.m
        barx = self._setParameterBarX(x)
        gamma = self._setParameterGamma(x)
        #print ["barx", barx]
        #print ["gamma", gamma]
        v = np.zeros(p+1)
        mat = np.zeros((p+1)*(p+1)).reshape(p+1,p+1)

        v[0] = barx[0]
        for i in range(p):
            v[i+1] = gamma[i+1][0]
        mat[0][0] = 1.0
        for i in range(p):
            mat[0][i+1] = barx[i+1]
            mat[i+1][0] = barx[i+1]
        for i in range(p):
            for j in range(p):
                mat[i+1][j+1] = gamma[i+1][j+1]
        return [mat,v]

class inflow:
    """ generate sampling for the class INARinfow """
    def __init__(self):
        return

    def getSample(self):
        return 1

class inflowPoisson(inflow):
    def __init__(self, lmd):
        self.lmd = lmd
        return

    def getSample(self):
        return np.random.poisson(self.lmd, 1)[0]

class inflowWithPrior(inflow):
    def __init__(self, prior):
        self.prior = prior
        return

    def getSample(self):
        return self.prior.getSample()

class inflowPoissonWithPrior(inflowWithPrior):
    def __init__(self, prior):
        self.prior = prior
        self.lmd = self.prior.getSample()
        return

    def getSample(self):
        self.lmd = self.prior.getSample()
        return np.random.poisson(self.lmd, 1)[0]

    def getLambda(self):
        return self.lmd

class prior:
    def __init__(self):
        return

    def getSample(self):
        return 1

class priorPareto(prior):
    def __init__(self, alpha):
        self.alpha = alpha
        self.mean = 1.0 / (alpha - 1.0)
        return

    def getSample(self):
        u = np.random.random_sample(1)
        return (1.0/(1.0 - u)**(1.0/self.alpha) - 1.0)


if __name__ == "__main__":
    ### INAR model without inflow.
    ### generate sample data.
    def INAR_test1(N):
        print "INAR_test1"
        inar = INAR(2)
        inar.r = np.array([0.1,0.8])
        inar._state = np.array([10*N,0])
        print inar.getSampleData(N)
        print ""
        return

    ### INAR model without inflow.
    ### generate sample data and evaluate the parameter r
    ### by a least square method.
    def INAR_test2(N):
        print "INAR_test2"
        inar = INAR(2)
        inar.r = np.array([0.1,0.8])
        inar._state = np.array([10*N,0])
        x = inar.getSampleData(N)
        inar.estimateParameters(x)
        print inar.r
        print ""
        return
    
    ### INAR model with constant inflow.
    ### generate sample data.
    def INARinflow_test1(N):
        print "INARinflow_test1"
        Inflow = inflow()
        inar = INARinflow(2, Inflow)
        inar.r = np.array([0.1,0.8])
        print inar.getSampleData(N)
        return

    ### INAR model with Poisson distributed inflows.
    ### generate sample data and evaluate the parameter r
    ### by a least square method.
    def INARinflow_test2(N):
        print "INARinflow_test2"
        Inflow = inflowPoisson(1.0)
        inar = INARinflow(2, Inflow)
        inar.r = np.array([0.1,0.8])
        x = inar.getSampleData(N)
        inar.estimateParameters(x)
        print inar.r
        print ""
        return

    ### excute the test functions ###

    INAR_test1(100)

    np.random.seed(100)
    INAR_test2(10000)
    INARinflow_test1(100)

    np.random.seed(100)
    INARinflow_test2(10000)

