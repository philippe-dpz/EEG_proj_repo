from .nefer import *

import math

# https://github.com/DynamicsAndNeuralSystems/catch22/tree/main/C  

def pow2(n : int) -> int :
    n -= 1

    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    return n + 1

def adjacent(y : Vector) -> Vector :
    return np.array(y[1 :]) - y[: -1]

def deviation(y : Vector) -> Vector :
    return np.array(y) - np.mean(y)

def stddev(y : Vector) -> float :
    return np.std(y)

def cov_mean(x : Vector, y : Vector) -> float :
    return np.dot(x, y) / len(x)

def boolInt(b : bool) -> int :
    return 2 - ~int(b)

def f_entropy(a : Vector) -> float :
    return -1 * sum([x * math.log(x) for x in a if (x > 0)])

def num_bins_auto(y : Vector) -> int :    
    if (stddev(y) < 0.001) : return 0
    
    return math.ceil((max(y) - min(y)) / (3.5 * stddev(y) / pow(len(y), 1 / 3)))

def histcounts_preallocated(y : Vector, nBins : int) -> tuple[Vector, Vector] : 
    size : int = len(y)
    # check min and max of input array
    minVal    = min(y)
    # if no number of bins given, choose spaces automatically
    if (nBins <= 0) : nBins = num_bins_auto(y)
    # and derive bin width from it
    binStep   = (max(y) - minVal) / nBins
    # variable to store counted occurances in
    binCounts = np.zeros(nBins)
    
    for i in range(size) :
        binInd = (y[i] - minVal) / binStep

        if (binInd < 0) : binInd = 0
        elif (binInd >= nBins) : binInd = nBins - 1 

        binCounts[int(binInd)] += 1

    binEdges = np.array(range(nBins + 1)) * binStep + minVal
    
    return binCounts, binEdges

def histBinAssign(y : Vector, binEdges : Vector) -> Vector :
    size : int = len(y)
    nEdges = range(len(binEdges))
    # variable to store counted occurances in
    binIdentity = np.zeros(size)

    for i in range(size) :
        # go through bin edges
        for j in nEdges :
            if (y[i] < binEdges[j]) :
                binIdentity[i] = j

                break
    
    return binIdentity

def corrs(x : Vector, y : Vector, size : int) -> float :
    dev_X = deviation(x[: size])
    dev_Y = deviation(y[: size])

    return sum(dev_X * dev_Y) / math.sqrt(sum(dev_X ** 2) * sum(dev_Y ** 2))

def autocorr_lag(x : Vector, lag : int) -> float :
    return corrs(x, x[lag :], len(x) - lag)

def fc_local_simple_mean(y : Vector, size : int) -> Vector :
    yest = [sum(y[i : i + size]) / size for i in range(len(y) - size)]

    return [y[i + size] - x for i, x in enumerate(yest)]

def twiddles(size : int) -> LComplex :
    return [np.exp(complex(0, x)) for x in -math.pi * np.array(range(size)) / size]

def _fft(a : LComplex, out : LComplex, tw : LComplex, step : int) -> None :
    size : int = len(a)
 
    if (step < size) :
        s2 = step << 1

        _fft(out, a, tw, s2)
        _fft(out[step :], a[step :], tw, s2)

        for i in range(0, size, step << 1) :
            t : complex = tw[i] * out[i + step]

            a[i >> 1] = out[i] + t
            a[(i + size) >> 1] = out[i] - t

def fft(y : LComplex, tw : LComplex) -> None :
    _fft(y, y.copy(), tw, 1)

def co_autocorrs(y : LComplex) -> Vector :
    nb   : int = len(y)
    nFFT : int = pow2(nb) << 1
    
    tw = twiddles(nFFT)
    F  = list(np.complex128(deviation(y))) \
       + [complex(0) for _ in range(nFFT - nb)]

    fft(F, tw)
    fft([f * f.conjugate() for f in F], tw)

    divisor = F[0]
    divisor /= divisor ** 2
    
    return (np.array(F) * divisor).real

def co_firstzero(y : Vector, max_tau : int) -> int :
    corrs = co_autocorrs(y)

    zeroCrossInd : int = 0

    while ((corrs[zeroCrossInd] > 0) & (zeroCrossInd < max_tau)) : zeroCrossInd += 1

    return zeroCrossInd

@deprecated("Problème dans l'implémentation")
def splinefit(y : Vector, nBreaks : int = 3, deg : int = 3) -> Vector :
    size      : int = len(y)
    piecesExt : int = 4
    nSpline   : int = 4
    pieces    : int = 2
    nCoeff    : int = nSpline * piecesExt

    breaks = [0, int(size >> 1) - 1, size - 1]
    
    # -- splinebase
    
    # repeat spacing
    hCopy = list(adjacent(breaks)) * 2
    # add breaks
    hExt  = adjacent(list(np.flip(breaks[0] - np.cumsum(hCopy[1 :]))) +
                     list(breaks) +
                     list(breaks[2] + np.cumsum(hCopy[: deg])))
    # expand h using the index matrix ii
    I2 = [[min(j + i, piecesExt - 1) for j in range(piecesExt)]
          for i in range(deg + 1)]
    # expanded h
    H  = [hExt[I2[i % nSpline][i // nSpline]] for i in range(nCoeff)]
    # recursive generation of B-splines
    Q  = np.zeros((nSpline, piecesExt))
    # initialise polynomial coefficients + 1
    Kf = np.zeros((nCoeff, nSpline))
    # Q2 = np.zeros((nCoeff, nSpline)) 

    for i in range(0, nCoeff, nSpline) : Kf[i][0] = 1

    for k in range(1, nSpline) :
        # antiderivatives of splines
        for j in range(k) :
            for l in range(nCoeff) : Kf[l][j] *= H[l] / (k - j)
        
        for l in range(nCoeff) :
            for m in range(nSpline) : Q[l % nSpline][l // nSpline] += Kf[l][m]
        
        # cumsum
        for l in range(piecesExt) :
            for m in range(1, nSpline) : Q[m][l] += Q[m - 1][l]
        
        for l in range(nCoeff) :
            md = l % nSpline 
            
            Kf[l][k] = 0 if (md == 0) else Q[md - 1][l // nSpline]  # questionable
        
        # normalise antiderivatives by max value
        fmax = [Q[nSpline - 1][i] for _ in range(nSpline) for i in range(piecesExt)]
        
        for j in range(k + 1) :
            for l in range(nCoeff) : Kf[l][j] /= fmax[l]

        # diff to adjacent antiderivatives
        for i in range(nCoeff - deg) :
            for j in range(k + 1) : Kf[i][j] -= Kf[deg + i][j]

        for i in range(1, nCoeff, nSpline) : Kf[i][k] = 0
    
    # scale coefficients
    scale = np.ones(nCoeff)
    
    for k in range(nSpline - 1) :
        scale = np.divide(scale, H)

        for i in range(nCoeff) :
            Kf[i][(nSpline - 1) - (k + 1)] *= scale[i]
    
    # reduce pieces and sort coefficients by interval number
    jj = [[nSpline * (j + 1) if (i == 0) else deg for j in range(pieces)]
          for i in range(nSpline)]
    
    for i in range(1, nSpline) :
        for j in range(pieces) : jj[i][j] += jj[i - 1][j]
    
    print([jj[i % nSpline][i // nSpline] - 2 for i in range(nSpline * pieces)])

    coefs_out = [[Kf[jj[i % nSpline][i // nSpline] - 2][j] for j in range(nSpline)]
                 for i in range(nSpline * pieces)]
    
    # -- create first B-splines to feed into optimization
    
    score : int = size * nSpline

    # x-values for B-splines
    xsB    = np.zeros(score)
    indexB = np.zeros(score)
    
    breakInd : int = 1

    for i in range(size) :
        if((i >= breaks[breakInd]) & (breakInd < nBreaks - 1)) : breakInd += 1
        
        m = breakInd - 1

        for j in range(nSpline) :
            p = i * nSpline + j

            xsB[p]    = i - breaks[m]
            indexB[p] = m * nSpline + j

    vB = [coefs_out[indexB[i]][0] for i in range(score)]
    
    for i in range(1, nSpline) :
        for j in range(score) :
            vB[j] = vB[j] * xsB[j] + coefs_out[indexB[j]][i]
    
    A = np.zeros((nSpline + 1) * size)

    breakInd = 0

    for i in range(score) :
        if (i / nSpline >= breaks[1]) : breakInd = 1

        A[(i % nSpline) + breakInd + (i // nSpline) * (nSpline + 1)] = vB[i]
    
    # coeffs of B-splines to combine by optimised weighting in x
    C = np.zeros((pieces + nSpline - 1, nSpline * pieces))

    n2 : int = nSpline << 1
    
    for i in range((nSpline ** 2) * pieces) :
        j = i // nSpline
    
        C[(i % nSpline) + (j % 2)][j] = coefs_out[i % n2][i // n2]
    
    x = np.linalg.solve(A, y) # lsqsolve_sub(nSpline + 1, A, y)
    
    # final coefficients
    coefsSpline = np.zeros((pieces, nSpline))
    
    # multiply with x
    for j in range(nSpline * pieces) :
        for i in range(nSpline + 1) :
            coefsSpline[j % pieces][j // pieces] += C[i][j] * x[i]
    
    # compute piecewise polynomial
    
    yOut = [coefsSpline[boolInt(i < breaks[1])][0] for i in range(size)]

    for i in range(1, nSpline) :
        for j in range(size) :
            p = boolInt(j < breaks[1])
            yOut[j] *= (j - breaks[1] * p) + coefsSpline[p][i]
    
    return yOut

def quantile(y : Vector, quant : float) -> float :
    size : int = len(y)
    tmp = y.copy()

    tmp.sort()
    
    q = .5 / size   # out of range limit?

    if (quant < q) : return tmp[0]              # min value
    if (quant > (1 - q)) : return tmp[size - 1] # max value
    
    quant_idx = size * quant - .5
     
    left  = int(math.floor(quant_idx)) 
    right = int(math.ceil(quant_idx))
    
    return tmp[left] + (quant_idx - left) * (tmp[right] - tmp[left]) / (right - left)

def sb_coarsegrain(y : Vector, how : str, num_groups : int) -> Vector :
    # if (how == "quantile") :
    #     raise ("ERROR in sb_coarsegrain: unknown coarse-graining method\n")
    
    ls = np.linspace(0, 1, num_groups + 1)
    th = [quantile(y, ls[i]) for i in range(num_groups + 1)]
    
    th[0] -= 1

    return [(i + 1) for j in range(len(y)) for i in range(num_groups) if (y[j] > th[i]) & (y[j] <= th[i + 1])]

def sb_binary_stats_diff_longstretch(y : Vector) :
    max_stretch = 0
    last        = 0
    size        = len(y)

    for i in range(size - 1) :
        if (y[i] | (i == size - 2)) :
            max_stretch = max(i - last, max_stretch)
            last = i
    
    return max_stretch

def linreg(x : Vector, y : Vector, n : int) -> tuple[int, float, float] : #, double* m, double* b
    xt = x[: n]
    yt = y[: n]

    sumx  = sum(xt)          #/* sum of x     */
    sumx2 = np.dot(xt, xt)   #/* sum of x**2  */
    denom = (n * sumx2 - sumx ** 2)

    # singular matrix. can't solve the problem.
    if (denom == 0) : return 1, 0, 0

    sumxy = np.dot(xt, yt)   #/* sum of x * y */
    sumy  = sum(yt)          #/* sum of y     */
    # sumy2 = np.dot(y, y)   #/* sum of y**2  */
    
    m = (n * sumxy - sumx * sumy) / denom
    b = (sumy * sumx2 - sumx * sumxy) / denom
    
    return 0, m, b

def welch(y : Vector, NFFT : int, Fs : float, window : Vector) -> tuple[int, Vector, Vector] : #, double ** Pxx, double ** f
    width : int = len(window)
    size  : int = len(y)
    # number of windows, should be 1
    ww    : int = width >> 1
    k     : int = int(size / ww) - 1

    dt = 1.0 / Fs
    df = 1.0 / (pow2(width) * dt)
    mn = np.mean(y)
    
    # fft variables
    P  = np.zeros(NFFT, dtype = complex)
    tw = twiddles(NFFT)

    for i in range(k) :
        # apply window
        xw = [window[j] * y[j + int(i * ww)] for j in range(width)]
        # initialise F 
        F  = [complex(x - mn) for x in xw] + [complex(0) for _ in range(width, NFFT)]
        
        fft(F, tw)

        for l in range(NFFT) : P[l] += F[l] ** 2
    
    Nout : int = (NFFT >> 1) + 1

    # normalising scale factor
    Pxx = P * k * dt / (np.linalg.norm(window) ** 2)

    Pxx[1 : Nout - 1] *= 2
    
    return Nout, Pxx[: Nout], np.array(range(Nout)) * df
