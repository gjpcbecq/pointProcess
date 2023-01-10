"""
Point Process Simulation

Classes and functions for point processes simulation, statistics computation and observation. 

Depends on numpy, scipy, matplotlib

Copyright 2013-2015, G. Becq, GIPSA-Lab, UMR 5216, Univ. Grenoble Alpes, CNRS;  F-38000 Grenoble, France. 
"""
"""
guillaume.becq@gipsa-lab.grenoble-inp.fr

This software is a computer program whose purpose is to compute directed information and causality measures on multivariates.

This software is governed by the CeCILL license under French law and abiding
by the rules of distribution of free software. You can use, modify and/ or
redistribute the software under the terms of the CeCILL license as circulated
by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and
redistribute granted by the license, users are provided only with a limited
warranty and the software's author, the holder of the economic rights, and the
successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with
loading, using, modifying and/or developing or reproducing the software by the
user in light of its specific status of free software, that may mean that it is
complicated to manipulate, and that also therefore means that it is reserved for
developers and experienced professionals having in-depth computer knowledge.
Users are therefore encouraged to load and test the software's suitability as
regards their requirements in conditions enabling the security of their systems
and/or data to be ensured and, more generally, to use and operate it in the same
conditions as regards security.  

The fact that you are presently reading this means that you have had knowledge
of the CeCILL license and that you accept its terms.
"""

#_______________________________________________________________________________
import numpy
import scipy.sparse
import matplotlib.pyplot
zeros = numpy.zeros

T_ITIME = 0
T_IPROCESS = 1
IS_X = 0b100
IS_T = 0b010
IS_K = 0b001

__all__ = ['IS_K', 'IS_T', 'IS_X', 'PP', 'PP_CoupledPoisson', 'PP_Poisson', 
    'PP_PoissonCoupled', 'PP_Truccolo', 'T_IPROCESS', 'T_ITIME', 
    'plotMinMax', 'simulateBernoulli', 'simulateCoupledPoisson', 'simulatePoisson', 'simulatePoissonContinuous', 'simulatePoissonCoupled', 'simulateTruccolo']
#_______________________________________________________________________________
#_______________________________________________________________________________
class PP(): 
    """
    Define a point process class
    
    A PP object contains methods for point process instances. 
    
    This is the parent of: 
    PP_Poisson
    PP_PoissonCoupled
    PP_Truccolo
    
    mode: binary number (xtk) 
     1st bit: time serie representation (x) 
     2nd bit: time couplet representation (t)
     3rd bit: sparse representation (k)
    
    """
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def __init__(self):
        """
        """
        self.nObs = 0
        self.nDim = 0
        self.model = "void object"
        self.param = "no parameter"
        self.x = zeros((self.nDim, self.nObs))
        self.mode = IS_X
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def __str__(self):
        """
        """
        message = 'an instance of {model:s} point process with parameters: \n'.format(model=self.model)
        message += str(self.param)
        return message
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def simulate(self, param=[]): 
        """
        Simulates the system according to the input u
        
        compute self.x, self.t or self.k
        
        """
        for i in range(self.nDim): 
            for j in range(self.nObs): 
                x[i, j] = 0
        self.x = x
        self.mode &= IS_X 
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def xToT(self):
        """
        x is a time serie representation
        x is an (nDim, nObs) array
        t is a continuous time representation
        t is an (2, nObs)
        k is discrete time representation
        sparse representation of x
        """
        x = self.x
        nDim = self.nDim
        nObs = self.nObs
        nEvtTotal = numpy.sum(x)
        # print("n evt total: " + str(nEvtTotal))
        t = zeros((2, nEvtTotal))
        k = -1
        for iObs in range(nObs): 
            for iDim in range(nDim): 
                nEvt = x[iDim, iObs]
                for iEvt in range(nEvt):
                    k += 1
                    # first sample is in [0, 1[
                    # we consider all events are at the same time
                    t[T_ITIME, k] = iObs 
                    # processes are numbered from 1 to nDim
                    t[T_IPROCESS, k] = iDim + 1
        self.t = t
        self.mode |= IS_T
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def xToK(self):
        """
        sparse representation of x 
        """
        k = scipy.sparse.lil_matrix(self.x)
        self.k = k
        self.mode |= IS_K
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def tToX(self): 
        """
        continuous time (2, nEvt) array to time serie (nObs, nDim) array representation
        process are numbered from 1 to nDim
        """
        t = self.t
        nDim = self.nDim
        nEvt = t.shape[1]
        # for example if 
        # t[T_ITIME, 0] = 0.12 with t[T_IPROCESS, 0] = 1
        # we must have x[0, 0] = 1
        nObs = numpy.max(t[T_ITIME, :]) + 1
        x = numpy.zeros((nDim, nObs))
        for iEvt in range(nEvt): 
            iObs = numpy.floor(t[T_ITIME, iEvt])
            iDim = t[T_IPROCESS, iEvt] - 1
            x[iDim, iObs] += 1
        self.x = x
        self.nObs = nObs
        self.mode |= IS_X
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def tToK(self):
        """
        time to sparse representation
        """
        t = self.t
        if ((self.mode & IS_X) == 0):
            self.tToX()
        x = self.x
        self.xToK()
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def kToX(self):
        """
        sparse to time serie representation
        """
        x = k.todense()
        self.x = x
        self.mode |= IS_X
        return 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def kToT(self):
        """
        sparse to time representation
        """
        k = self.k
        x = kToX(k)
        self.x = x
        t = XToT(X)
        self.t = t
        self.mode |= (IS_X | IS_T)
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def toX(self): 
        """
        to time serie representation
        """
        if (self.mode & IS_T) != 0:
            self.tToX()
            return None
        if (self.mode & IS_K) != 0: 
            self.kToX()
            return None
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def toT(self): 
        """
        to time representation
        """
        if (self.mode & IS_X) != 0:
            self.xToT()
            return None
        if (self.mode & IS_K) != 0: 
            self.kToT()
            return None
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def rasterPlot(self):
        """
        Raster plot
        
        """
        nDim = self.nDim
        nObs = self.nObs
        if (self.mode & IS_T == 0): 
            self.toT()
        if (self.mode & IS_X == 0): 
            self.toX()
        t = self.t
        x = self.x
        matplotlib.pyplot.subplot(2, 1, 1)
        matplotlib.pyplot.plot(t[T_ITIME, :], t[T_IPROCESS, :], '.')
        matplotlib.pyplot.axis([0, nObs, 0, nDim + 1])
        matplotlib.pyplot.yticks(range(1, nDim + 1))
        matplotlib.pyplot.title('Raster Plot')
        matplotlib.pyplot.ylabel('iProcess (#)')
        matplotlib.pyplot.xlabel('Samples (#)')
        matplotlib.pyplot.subplot(2, 1, 2)
        plotMinMax(numpy.array(x, 'float'))
        matplotlib.pyplot.axis([0, nObs, 0, nDim + 1])
        matplotlib.pyplot.yticks(range(1, nDim + 1))
        matplotlib.pyplot.title('Variation of the Number of Events per Samples')
        matplotlib.pyplot.ylabel('iProcess (#)')
        matplotlib.pyplot.xlabel('Samples (#)')
        matplotlib.pyplot.tight_layout()
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def plot_counting(): 
        """
        plot the counting processes
        """
        cumX = numpy.cumsum(x, axis=1) 
        plotMinMax(numpy.array(cumX, 'float'))
        matplotlib.pyplot.axis([0, nObs, 0, nDim + 1])
        matplotlib.pyplot.yticks(range(1, nDim + 1))
        matplotlib.pyplot.title('Counting Process (Number of Events)')
        matplotlib.pyplot.ylabel('iProcess (#)')
        matplotlib.pyplot.xlabel('Samples (#)')
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def plot(self):
        """
        same as rasterPlot
        """
        self.rasterPlot()
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def interSpikeHistogram(self):
        """
        inter spike interval
        
        """
        if (self.mode & IS_T == 0): 
            self.toT()
        nDim = self.nDim
        nEvt = t.shape[1]
        resIsi = []
        if (nEvt == 0): 
            return resIsi
        timeEVT = zeros((nDim, nEvt), 'float')
        diffTimeEVT = zeros((nDim, nEvt), 'float')
        k = zeros((nDim, )) - 1 # start at -1 to get 1st index at 0
        for iEvt in range(nEvt):
            iDim = t[T_IPROCESS, iEvt] - 1 # -1 for indexing for example process 1 is indexed to 0
            iTime = t[T_ITIME, iEvt]
            k[iDim] += 1
            timeEVT[iDim, k[iDim]] = iTime
        diffTimeEvt = numpy.diff(timeEVT, axis=1)
        print('diff time evt: '+ str(diffTimeEvt.shape))
        for iDim in range(nDim): 
            resIsi.append(diffTimeEvt[iDim, :k[iDim]])
        return resIsi
    isi = ISI = interSpikeHistogram
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def plotIsi(self, resIsi, nBin=10): 
        """
        Plot inter spike intervals
        
        """
        nDim = len(resIsi)
        for iDim in range(nDim): 
            matplotlib.pyplot.subplot(nDim, 1, iDim+1)
            matplotlib.pyplot.hist(resIsi[iDim], nBin)
            matplotlib.pyplot.title('Process ' + str(iDim))
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def stats(self): 
        """
        compute basic statistics from x 
        
        return the tuple (xMean, xVar, xF) with: 
        xMean: mean of x
        xVar: variance of x
        xF: Fano factor of x
        """
        if (self.mode & IS_X == 0):
            self.toX()
        x = self.x
        xMean = numpy.mean(x, axis=1)
        xVar = numpy.var(x, axis=1)
        xF = xMean / xVar
        return (xMean, xVar, xF) 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def jointIntervalDensity(self, resIsi):
        """
        Plot the joint interval density
        
        jointIntervalDensity(resISI)

        """
        nDim = len(resIsi)
        for iDim in range(nDim): 
            matplotlib.pyplot.subplot(nDim, 1, iDim+1)
            matplotlib.pyplot.plot(resIsi[iDim][:-1], resIsi[iDim][1:], '.')
            matplotlib.pyplot.title('Process ' + str(iDim))
            matplotlib.pyplot.xlabel('ISI (k)')
            matplotlib.pyplot.ylabel('ISI (k+1)')
        matplotlib.pyplot.tight_layout()
        return None
    JID = jointIntervalDensity
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def crossIntervalHistogram(self, maxLag=10): 
        """
        Cross Interval Histogram  
        
        """
        if (self.mode & IS_X == 0): 
            self.toX()
        x = self.x 
        (nDim, nObs) = (self.nDim, self.nObs)
        if (maxLag > nObs): 
            maxLag = nObs
        xIsi = numpy.zeros((2 * maxLag + 1, nDim, nDim))
        for iDim in range(nDim):
            for jDim in range(nDim):
                for iObs in range(nObs - maxLag):
                    if (x[iDim, iObs] > 0):
                        for jObs in range(iObs, iObs + maxLag + 1):
                            if (x[jDim, jObs] > 0): 
                                # r(i, j)(k) = f(x_i[n+k], x_j[n])
                                # here we do f(x_j[n+k], x_i[n]) 
                                iLag = maxLag + jObs - iObs
                                xIsi[iLag, jDim, iDim] += 1
                for iObs in range(-1, -(nObs - maxLag), -1):
                    if (x[iDim, iObs] > 0):
                        for jObs in range(iObs - 1, iObs - maxLag - 1, -1):
                            if (x[jDim, jObs] > 0):
                                # r(i, j)(-k) = f(x_i[n-k], x_j[n])
                                # here we do f(x_j[n-k], x_i[n]) 
                                iLag = maxLag + jObs - iObs
                                xIsi[iLag, jDim, iDim] += 1
        nTick = 5
        stepX = int(numpy.floor((2 * maxLag + 1) / (nTick - 1)))
        for iDim in range(nDim):
            for jDim in range(nDim):
                matplotlib.pyplot.subplot(nDim, nDim, nDim * iDim + jDim + 1)
                matplotlib.pyplot.plot(xIsi[:, iDim, jDim], '.-')
                matplotlib.pyplot.title(str(iDim) + '/' + str(jDim))
                matplotlib.pyplot.xticks(range(0, 2 * maxLag + 1, stepX), 
                    range(-maxLag, maxLag+1, stepX))
                if (iDim == (nDim - 1)) & (jDim == 0): 
                    matplotlib.pyplot.ylabel('number of events (#)')
                    matplotlib.pyplot.xlabel('sample (#)')
        matplotlib.pyplot.tight_layout()
        return None                
    XISI = crossIntervalHistogram
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
#_______________________________________________________________________________
#_______________________________________________________________________________
class PP_Poisson(PP): 
    """
    Poisson process
    
    lam: 1d array containing lambda values for the nDim processes. 
    
    """
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def __init__(self, nObs, lam):
        """
        pp1 = PP_Poisson(nObs, lam)
        """
        self.nObs = nObs
        self.nDim = lam.shape[0]
        self.model = "Poisson Process"
        self.lam = lam
        self.simulate(nObs)
        self.param = lam 
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def simulate(self, nObs):
        """
        use simulatePoisson with nObs samples
        """
        self.nObs = nObs
        self.x = simulatePoisson(nObs, self.lam)
        self.mode = IS_X
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
#_______________________________________________________________________________
#_______________________________________________________________________________
class PP_PoissonCoupled(PP): 
    """
    Coupled Poisson process
    
    u: (nDim, nObs) array, input on the system
    A: (m, nDim, nDim) array, coupling with delays 
    
    """
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def __init__(self, u, A):
        """
        pp1 = PP_PoissonCoupled(u, A)
        """
        self.nObs = u.shape[1]
        self.nDim = u.shape[0]
        self.m = A.shape[0]
        self.model = "Coupled Poisson Process"
        self.A = A
        self.u = u
        self.simulate(u)
        self.param = A
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def simulate(self, u):
        """
        use simulatePoissonCoupled on input u
        """
        self.u = u
        (self.x, self.lam) = simulatePoissonCoupled(u, self.A)
        self.mode = IS_X
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
#_______________________________________________________________________________
#_______________________________________________________________________________
class PP_Truccolo(PP): 
    """
    Truccolo Poisson process
    
    u: (nDim, nObs) array, extrinsec values, inputs on the system 
    c: (nR, nDim) array, gamma, autoregressive coefficients
    b: (nQ, nDim) array, beta, ensemble influence
    
    """
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def __init__(self, u, c, b):
        """
        pp1 = PP_Truccolo(u, c, b)
        """
        self.nDim = u.shape[0]
        self.nObs = u.shape[1]
        self.Q = c.shape[0]
        self.R = b.shape[0]
        self.model = "Truccolo model for Poisson Processes"
        self.c = c
        self.b = b
        self.u = u
        self.simulate(u)
        self.param = [c, b]
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    def simulate(self, u):
        """
        use simulateTruccolo on input u
        """
        self.u = u
        self.nObs = u.shape[1]
        (self.x, self.lam) = simulateTruccolo(u, self.c, self.b)
        self.mode = IS_X
        return None
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
#_______________________________________________________________________________
#_______________________________________________________________________________
def simulateBernoulli(nObs, p): 
    """
    Simulates Bernoulli processes
    
    Syntax
    
    x = simulateBernoulli(nObs, p)
    
    Input
    
    nObs: number of samples to simulate
    p: (nDim, ) 1d array, contains the probabilities for nDim processes 
    
    Output
    
    x: (nDim, nObs) array with integer values. 
     It contains one event observed at each sample
    
    Description

    For each sample the number of events follow a Bernoulli distribution 
    
    $$ Pr(x(t) = 1) = p $$ 

    Note that for p < 0.3, the process looks like a Poisson process with lambda = p
    
    Example
    
    >>> random.seed(0)
    >>> nObs = 100
    >>> lam = array([0.1])
    >>> x = simulateBernoulli(nObs, lam)
    >>> print(x)
    >>> print(mean(x))
    >>> print(var(x))
    
    [[1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 0 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1
     1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1]]
    0.85
    0.1275
    
    """
    nDim = p.shape[0]
    x = zeros((nDim, nObs), dtype='int')
    for iProcess in range(nDim): 
        x[iProcess, :] = numpy.random.binomial(1, p[iProcess], nObs)
    return x
#_______________________________________________________________________________
#_______________________________________________________________________________
def simulatePoisson(nObs, lam): 
    """
    Simulates Poisson processes
    
    Syntax
    
    x = simulatePoisson(nObs, lam)
    
    Input
    
    nObs: number of samples to simulate
    lam: (nDim, ) 1d array, contains the rate lambda for nDim processes 
    
    Output
    
    x: (nDim, nObs) array with integer values. 
     It contains the number of events observed at each sample
    
    Description

    For each sample the number of events follow a Poisson distribution 
    
    $$ Pr(x(t) = N) = \frac{\lambda^N}{N!} \, \exp{-\lambda} $$ 

    Note that for lambda < 0.3, x is rarely > 1 and the process looks like a Bernoulli process. 
    
    Example
    
    >>> random.seed(0)
    >>> nObs = 100
    >>> lam = array([0.1])
    >>> x = simulatePoisson(nObs, lam)
    >>> print(x)
    >>> print(mean(x))
    >>> print(var(x))
    
    [[0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1]]
    0.11
    0.0979
    
    Example
    
    >>> random.seed(0)
    >>> nObs = 100
    >>> lam = array([0.8])
    >>> x = pp.simulatePoisson(nObs, lam)
    >>> print(x)
    >>> print(mean(x))
    >>> print(var(x))
    
    [[1 1 0 1 2 1 2 0 0 4 1 0 1 2 0 1 1 1 2 0 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 1 0
      0 1 1 2 1 0 0 0 0 0 0 0 0 1 0 1 2 1 1 0 1 1 1 2 1 1 1 2 1 3 1 2 1 0 1 1 0
      0 1 1 1 1 0 3 3 1 1 2 0 0 1 4 0 1 1 0 0 1 0 2 0 1 0]]
    0.84
    0.7744
    
    """
    nDim = lam.shape[0]
    x = zeros((nDim, nObs), dtype='int')
    for iProcess in range(nDim): 
        x[iProcess, :] = numpy.random.poisson(lam[iProcess], size=nObs)
    return x
#_______________________________________________________________________________
#_______________________________________________________________________________
def simulatePoissonCoupled(u, A): 
    """
    Simulate coupled inhomogenous Poisson processes. 
    
    Syntax
    
    x = simulatePoissonCoupled(u, A)    
    
    Input 
    
    A: (m, nDim, nDim) array of couplings of the ndim point process for m delays. 
    u: (nDim, nObs) array, input on the lambda
    
    Output
    
    x: (nDim, nObs) array with integer values, number of events observed. 
    
    Description
    
    Generate Poisson variate at each sample with varying rate such that: 
    
    $$ lamda_i(n) = \sum_{k = 1}^{m} \sum_{j = 1}^{nDim} a_{i, j, k} \, lambda_j(n - k) + u(n) $$
    
    Example
    
    >>> random.seed(0)
    >>> nDim = 3
    >>> nObs = 10
    >>> A = array([[[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 1.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            ])
    >>> u = zeros((nDim, nObs))
    >>> u[1, :] = 0.01 + 0.99 * random.rand(nObs)
    >>> u[2, :] = 0.1 + 0.09 * sin(2 * pi * 0.01 * arange(nObs))
    >>> (N, lam) = pp.simulatePoissonCoupled(u, A)
    >>> print(N)
    >>> print(around(lam, 2))
    
    [[0 0 0 0 0 1 0 1 0 0]
     [0 0 1 1 1 0 0 1 1 0]
     [0 0 0 0 0 0 0 0 0 1]]
    [[ 0.    0.    0.    0.    0.11  0.12  0.12  0.13  0.13  0.14]
     [ 0.    0.    0.61  0.55  0.43  0.65  0.44  0.89  0.96  0.39]
     [ 0.    0.    0.11  0.12  0.12  0.13  0.13  0.14  0.14  0.15]]
    
    """
    (nDim, nObs) = u.shape 
    m = A.shape[0]
    lam = zeros((nDim, nObs))
    x = zeros((nDim, nObs), dtype='int')
    # print(lam)
    # print(x)
    # print(s0)
    print(u.shape)
    for iObs in range(m, nObs): 
        s = zeros((nDim,))
        for k in range(0, m):
            s1 = numpy.dot(A[k, :, :], lam[:, iObs - (k + 1)])
            s += s1
        lam[:, iObs] = s + u[:, iObs]
        # lam[(lam[:, iObs] < 0), iObs] = 0
        x[:, iObs] = simulatePoisson(1, lam[:, iObs])[:, 0]
    return (x, lam)       
#_______________________________________________________________________________
#_______________________________________________________________________________
def simulateTruccolo(u, c, b):
    """
    Simulates multivariate point processes describes in Truccolo et al. 2005
    
    (N, l) = simulateTruccolo(u, c, b)
    
    Input
    
    nObs: number of samples to compute. 
    c: (nDim, nQ) matrix of the autoregressive process. 
    b: (W, nDim, nDim) matrix of the ensemble influence. 
    u: extrinsic values applied to the model. 
    
    Output
    
    N : (nDim, nObs) integer time series. 
    l : (nDim, nObs) conditional intensity function
    
    See Truccolo et al. 
    
    c: gamma
    b: beta
    
    Example
    
    >>> random.seed(0)
    >>> nDim = 3
    >>> nObs = 10
    >>> c = array([[-3., 0., 0.], [-3., 0., 0.], [-3., 0., 0.]])
    >>> b = array([[[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            [[0, 0., 10.], [0., 0., 0.], [0., 0., 0.]], 
    >>>            ])
    >>> u = zeros((nDim, nObs))
    >>> u[1, :] = random.rand(nObs)
    >>> u[2, :] = log(10 + 9.9 * sin(2 * pi * 0.01 * arange(nObs)))
    >>> (N, lam) = simulateTruccolo(u, c, b)
    >>> print(N)
    >>> print(around(lam))
    
    [[0 0 0 0 0 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 1 0]
     [0 0 0 0 3 1 0 1 2 0]]
    [[ 0.    0.    0.05  0.05  0.05  1.    0.01  0.02  0.14  0.14]
     [ 0.    0.    0.09  0.07  0.03  0.12  0.03  0.52  0.8   0.02]
     [ 0.    0.    0.56  0.59  0.62  0.65  0.68  0.71  0.74  0.76]]

    """
    nQ = c.shape[1]
    nR = b.shape[0]
    nDim = u.shape[0]
    nObs = u.shape[1]
    N = zeros((nDim, nObs), 'int')
    DN = zeros((nDim, nObs), 'int')
    l = zeros((nDim, nObs), 'float')
    nMax = numpy.max([nQ, nR])
    print(nMax)
    for iObs in range(nMax, nObs): 
        #print("_______________________________________")
        #print("iObs: " + str(iObs))
        for iDim in range(nDim):
            #   print('==iDim=='+str(iDim))
            # Autoregressive influence
            logLiAutoregressive = c[iDim, 0]
            for iLag in range(1, nQ): 
                logLiAutoregressive += (c[iDim, iLag] * DN[iDim, iObs - iLag])
            # Ensemble influence: integration with delay
            logLiEnsemble = b[0, iDim, iDim]
            for jDim in range(nDim): 
                if (jDim == iDim): 
                    pass
                else: 
                    for iLag in range(1, nR): 
                        # print('iLag: ' + str(iLag))
                        # print("bj: " + str(bj[iLag, :]))
                        # print("xW: " + str(xW[:, iObs - iLag]))
                        logLiEnsemble += b[iLag, iDim, jDim] * (
                            DN[jDim, iObs - iLag])
                        # The proposition below, proposed in the article, 
                        # can not work properly with this simulation since
                        # the instantaneous sample x[iDim, 0] is necessary
                        # and modified by process order. 
                        # logLiEnsemble += bj[iLag, iDim] * (
                        #    N[iDim, iObs - (iLag - 1) * W] - 
                        #    N[iDim, iObs - iLag * W])
                        # print("log: " + str(logLiEnsemble))
            # Extrinsic covariate X: 
            logLiExtrinsic = u[iDim, iObs]
            # All influences
            logLi = (logLiAutoregressive + logLiEnsemble + logLiExtrinsic)
            l[iDim, iObs] = numpy.exp(logLi)
        # saturate lambda at 10 to avoid overflow. 
        c1 = (l[:, iObs] > 10)
        l[c1, iObs] = 10
        N[:, iObs] = simulatePoisson(1, l[:, iObs]).squeeze()
        DN[:, iObs] = N[:, iObs] - N[:, iObs - 1]
    return (N, l)
#_______________________________________________________________________________
#_______________________________________________________________________________
def simulatePoissonContinuous(T, lam, nMax=1000):
    """
    Simulate a continuous Poisson process by the method of thinning. 
    
    Syntax
    
    t = simulatePoissonContinuous(T, lam, nMax=1000)
    
    Input 
    
    T: float, maximum time
    lam: (nDim, ) array of lambda parameters
    nMax: int, maximal number of events 
    
    Output
    
    t: (nEvt, 2) array, instant and process indices of the events
    
    Description
    
    Compute instants for events of homogeneous and independent Poisson processes using the thinning method. 
    
    For coupled or inhomogenous Poisson processes, it is easier to get samples using simulatePoisson or simulatePoissonCoupled, setting the lambda suffisantly low and consider a high sampling rate. 
    
    Example
    
    >>> numpy.random.seed(0)
    >>> T = 10
    >>> lam = array([1., 0.5, 3.])
    >>> t = pp.simulatePoissonContinuous(T, lam, nMax=1000)
    >>> print(around(t[0], 2))
    >>> print(t[1])
    
    [ 0.61  1.89  2.75  2.91  8.27  0.8   1.18  5.16  6.27  6.5   9.63  0.74
      1.25  1.25  1.32  1.91  1.98  2.41  2.69  2.71  2.78  3.14  3.72  4.05
      5.77  6.23  6.31  7.7   7.87  8.04  8.79]
    [ 1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  3.  3.  3.  3.  3.  3.  3.
      3.  3.  3.  3.  3.  3.  3.  3.  3.  3.  3.  3.  3.]
  
    """
    # check Martin Haugh's Lesson (pdf article)
    nDim = lam.shape[0]
    t = numpy.zeros((2, nMax))
    i = 0
    for iDim in range(nDim): 
        ti = 0
        u = numpy.random.rand(1)
        ti = ti - numpy.log(u) / lam[iDim]
        while (ti < T): 
            if (i < nMax):
                i += 1
                t[T_ITIME, i - 1] = ti
                t[T_IPROCESS, i - 1] = iDim + 1
                u = numpy.random.rand(1)
                ti = ti - numpy.log(u) / lam[iDim]
            else : 
                return t[:, :i]
    return t[:, :i]
#_______________________________________________________________________________
#_______________________________________________________________________________
def plotMinMax(x): 
    """
    Plot the succession of signals with preprocessing using Min and Max values.
    
    Syntax
    
    y = plotMinMax(x)
    
    Input 
    
    x: (nDim, nObs) ndarray
    
    Output 
    
    y: (nDim, nObs) ndarray
    
    Example
    
    >>> numpy.random.seed(0)
    >>> x = numpy.random.randn(3, 10)
    >>> y = pp.plotMinMax(x) 
    >>> print(numpy.around(y, 2))    

    [[ 1.28  0.94  1.09  1.4   1.31  0.6   1.08  0.81  0.82  0.95]
     [ 1.94  2.39  2.15  1.93  2.04  2.    2.4   1.82  2.    1.6 ]
     [ 2.6   3.13  3.17  2.9   3.4   2.78  3.03  2.99  3.28  3.27]]

    
    """
    assert(x.ndim == 2)
    (nDim, nObs) = x.shape
    y = numpy.empty((nDim, nObs))
    for iDim in range(nDim):
        xi = x[iDim, :]
        yi = y[iDim, :]
        minX = min(xi)
        maxX = max(xi)
        rX = maxX - minX
        if (rX == 0): 
            yi = 0
        else : 
            yi = (xi - minX) / rX - 0
        yi *= 0.8
        yi += iDim + 0.5 + 0.1
        y[iDim, :] = yi
    matplotlib.pyplot.plot(y.T)
    return y
#_______________________________________________________________________________