from .nefer import *

import math

# https://github.com/DynamicsAndNeuralSystems/catch22/tree/main/C  

def ceil_pow2(n : int) -> int :
    n -= 1

    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    return n + 1

def bool_int(b : bool) -> int :
    return ~int(b) + 2

def adjacent(y : Vector) -> Vector :
    return np.array(y[1 :]) - y[: -1]

def deviation(y : Vector) -> Vector :
    return np.array(y) - np.mean(y)

def complex_padding(y : Vector, size : int) -> LComplex :
    return np.append(np.complex128(y), np.zeros(size - len(y), dtype = complex))

def stddev(y : Vector) -> float :
    return np.std(y)

def cov_mean(x : Vector, y : Vector) -> float :
    return np.dot(x, y) / len(x)

def f_entropy(y : Vector) -> float :
    return -1 * sum([x * math.log(x) for x in y if (x > 0)])

def num_bins_auto(y : Vector) -> int :
    tmp = np.std(y)
    
    if (tmp < 0.001) : return 0
    
    m_y = (max(y) - min(y)) * len(y) ** (1 / 3)

    # return int(math.ceil(m_y / (3.5 * tmp / (len(y) ** (1 / 3)))))
    return int(math.ceil(m_y / (3.5 * tmp)))

def histcounts_preallocated(y : Vector, num_bins : int) -> tuple[Vector, Vector] : 
    size : int = len(y)
    # check min and max of input array
    min_val    = min(y)
    # if no number of bins given, choose spaces automatically
    if (num_bins <= 0) : num_bins = num_bins_auto(y)
    # and derive bin width from it
    bin_step   = (max(y) - min_val) / num_bins
    # variable to store counted occurances in
    bin_counts = np.zeros(num_bins)
    
    for i in range(size) :
        bin_ind = (y[i] - min_val) / bin_step

        if (bin_ind < 0) : bin_ind = 0
        elif (bin_ind >= num_bins) : bin_ind = num_bins - 1 

        bin_counts[int(bin_ind)] += 1

    bin_edges = np.arange(num_bins + 1) * bin_step + min_val
    
    return bin_counts, bin_edges

def histBinAssign(y : Vector, bin_edges : Vector) -> Vector :
    size    : int = len(y)
    n_edges : int = range(len(bin_edges))
    # variable to store counted occurances in
    res = np.zeros(size)

    for i in range(size) :
        # go through bin edges
        for j in n_edges :
            if (y[i] < bin_edges[j]) :
                res[i] = j

                break
    
    return res

def histcount_edges(y : Vector, bin_edges : Vector) -> Vector :
    size    : int = len(y)
    n_edges : int = range(len(bin_edges))
    # variable to store counted occurances in
    res = np.zeros(size)

    for i in range(size) :
        # go through bin edges
        for j in n_edges :
            if (y[i] <= bin_edges[j]) :
                res[i] = j

                break
    
    return res

def corrs(x : Vector, y : Vector, size : int) -> float :
    dev_X = deviation(x[: size])
    dev_Y = deviation(y[: size])
    den   = np.dot(dev_X, dev_X) * np.dot(dev_Y, dev_Y)

    return np.dot(dev_X, dev_Y) / math.sqrt(den)

def autocorr_lag(x : Vector, lag : int) -> float :
    return corrs(x, x[lag :], len(x) - lag)

def fc_local_simple_mean(y : Vector, train_length : int) -> Vector :
    yest = [sum(y[i : i + train_length]) / train_length for i in range(len(y) - train_length)]

    return y[train_length :] - np.array(yest) # return [y[i + size] - x for i, x in enumerate(yest)]

def twiddles(size : int) -> LComplex :
    k = -math.pi / size

    return np.exp([complex(0, x) for x in np.arange(size) * k])

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

def co_autocorrs(y : Vector) -> Vector :
    nb   : int = len(y)
    nFFT : int = ceil_pow2(nb) << 1

    tw = twiddles(nFFT)
    F  = complex_padding(deviation(y), nFFT)

    fft(F, tw)

    F *= np.conjugate(F)

    fft(F, tw)

    divisor = F[0]

    return (np.array(F) / divisor).real

def co_firstzero(y : Vector, max_tau : int) -> int :
    corrs = co_autocorrs(y)
    signs = np.sign(corrs)

    for i in range(min(len(signs) - 1, max_tau)) :
        if (signs[i] != signs[i + 1]) : return i + 1

    zero_cross_ind : int = 0

    while ((corrs[zero_cross_ind] > 0) & (zero_cross_ind < max_tau)) :
        zero_cross_ind += 1

    return zero_cross_ind

def lsqsolve_sub(A, y : Vector) -> Vector :
    AT  = A.T

    return np.linalg.solve(np.matmul(AT, A), np.matmul(AT, y))

def splinefit(y : Vector, deg : int = 3) -> Vector :
    size       : int = len(y)
    pieces_ext : int = 4
    n_breaks   : int = 3
    n_spline   : int = 4
    pieces     : int = 2
    n_coeff    : int = n_spline * pieces_ext

    breaks = [0, int(size >> 1) - 1, size - 1]
    
    # -- splinebase
    
    # repeat spacing
    hCopy = list(adjacent(breaks)) * 2
    # add breaks
    hExt  = adjacent(list(np.flip(breaks[0] - np.cumsum(hCopy[1 :]))) +
                     list(breaks) +
                     list(breaks[2] + np.cumsum(hCopy[: deg])))
    # expand h using the index matrix ii
    I2 = [[min(j + i, pieces_ext - 1) for j in range(pieces_ext)]
          for i in range(deg + 1)]
    # expanded h
    H  = [hExt[I2[i % n_spline][i // n_spline]] for i in range(n_coeff)]
    # recursive generation of B-splines
    Q  = np.zeros((n_spline, pieces_ext))
    # initialise polynomial coefficients
    Kf = np.zeros((n_coeff + 1, n_spline))

    for i in range(0, n_coeff, n_spline) : Kf[i][0] = 1

    for k in range(1, n_spline) :
        # antiderivatives of splines
        for j in range(k) :
            for l in range(n_coeff) : Kf[l][j] *= H[l] / (k - j)
        
        for l in range(n_coeff) :
            for m in range(n_spline) : Q[l % n_spline][l // n_spline] += Kf[l][m]
        
        # cumsum
        for l in range(pieces_ext) :
            for m in range(1, n_spline) : Q[m][l] += Q[m - 1][l]
        
        for l in range(n_coeff) :
            md = l % n_spline 
            
            Kf[l][k] = 0 if (md == 0) else Q[l % n_spline - 1][l // n_spline]  # questionable
        
        # normalise antiderivatives by max value
        fmax = [Q[n_spline - 1][i] for _ in range(n_spline) for i in range(pieces_ext)]
        
        for j in range(k + 1) :
            for l in range(n_coeff) : Kf[l][j] /= fmax[l]

        # diff to adjacent antiderivatives
        for i in range(n_coeff - deg) :
            for j in range(k + 1) : Kf[i][j] -= Kf[deg + i][j]

        for i in range(1, n_coeff, n_spline) : Kf[i][k] = 0
    
    # scale coefficients
    scale = np.ones(n_coeff)
    
    for k in range(n_spline - 1) :
        scale = np.divide(scale, H)

        for i in range(n_coeff) :
            Kf[i][(n_spline - 1) - (k + 1)] *= scale[i]
    
    # reduce pieces and sort coefficients by interval number
    jj = [[n_spline * (j + 1) if (i == 0) else deg for j in range(pieces)]
          for i in range(n_spline)]
    
    for i in range(1, n_spline) :
        for j in range(pieces) : jj[i][j] += jj[i - 1][j]

    coefs_out = [[Kf[jj[i % n_spline][i // n_spline] - 1][j] for j in range(n_spline)]
                 for i in range(n_spline * pieces)]
    
    # -- create first B-splines to feed into optimization
    
    n_size : int = size * n_spline

    # x-values for B-splines
    xsB    = np.zeros(n_size, dtype = int)
    indexB = np.zeros(n_size, dtype = int)
    
    stop : int = 1

    for i in range(size) :
        if((i >= breaks[stop]) & (stop < n_breaks - 1)) : stop += 1
        
        m = stop - 1

        for j in range(n_spline) :
            p = i * n_spline + j

            xsB[p]    = i - breaks[m]
            indexB[p] = m * n_spline + j

    vB = [coefs_out[indexB[i]][0] for i in range(n_size)]
    
    for i in range(1, n_spline) :
        for j in range(n_size) :
            vB[j] = vB[j] * xsB[j] + coefs_out[indexB[j]][i]
    
    A = np.zeros((n_spline + 1) * size)

    stop = 0

    for i in range(n_size) :
        if (i / n_spline >= breaks[1]) : stop = 1

        A[stop + (i % n_spline) + (i // n_spline) * (n_spline + 1)] = vB[i]
    
    A[-1] = 1
    
    x = lsqsolve_sub(np.reshape(A, (size, 5)), y)

    # coeffs of B-splines to combine by optimised weighting in x
    C = np.zeros((pieces + n_spline - 1, n_spline * pieces))

    n2 : int = n_spline << 1
    
    for i in range((n_spline ** 2) * pieces) :
        j = i // n_spline
    
        C[(i % n_spline) + (j % 2)][j] = coefs_out[i % n2][i // n2]
    
    # final coefficients
    coefs_spline = np.zeros((pieces, n_spline))
    
    # multiply with x
    for j in range(n_spline * pieces) :
        for i in range(n_spline + 1) :
            coefs_spline[j % pieces][j // pieces] += C[i][j] * x[i]
    
    y_out = [coefs_spline[~int(i < breaks[1]) + 2][0] for i in range(size)]

    for i in range(1, n_spline) :
        for j in range(size) :
            p = ~int(j < breaks[1]) + 2
            y_out[j] *= (j - breaks[1] * p) + coefs_spline[p][i]
    
    return y_out

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

def sb_coarsegrain(y : Vector, num_groups : int) -> Vector :
    # if (how == "quantile") :, how : str
    #     raise ("ERROR in sb_coarsegrain: unknown coarse-graining method\n")
    n1 : int = num_groups + 1
    ls = np.linspace(0, 1, n1)
    th = [quantile(y, ls[i]) for i in range(n1)]
    
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

def linreg(x : Vector, y : Vector, size : int) -> tuple[int, float, float] :
    xt = x[: size]
    yt = y[: size]

    sumx  = sum(xt)          #/* sum of x     */
    sumx2 = np.dot(xt, xt)   #/* sum of x**2  */
    denom = (size * sumx2 - sumx ** 2)

    # singular matrix. can't solve the problem.
    if (denom == 0) : return 1, 0, 0

    sumxy = np.dot(xt, yt)   #/* sum of x * y */
    sumy  = sum(yt)          #/* sum of y     */
    # sumy2 = np.dot(y, y)   #/* sum of y**2  */
    
    m = (size * sumxy - sumx * sumy) / denom
    b = (sumy * sumx2 - sumx * sumxy) / denom
    
    return 0, m, b

def welch(y : Vector, NFFT : int, Fs : float, window : Vector) -> tuple[int, Vector, Vector] :
    size  : int = len(y)
    width : int = len(window)
    # number of windows, should be 1
    ww    : int = width >> 1
    k     : int = int(size / ww) - 1

    dt = 1.0 / Fs
    
    # fft variables
    P  = np.zeros(NFFT, dtype = float)
    tw = twiddles(NFFT)

    for i in range(k) :
        gap = i * ww
        res = window * np.array(y[gap : width + gap])
        # apply window / initialise F 
        F   = complex_padding(deviation(res), NFFT)
    
        fft(F, tw)

        P += abs(F) ** 2
    
    n_out : int = (NFFT >> 1) + 1

    # normalising scale factor
    Pxx = P * k * dt / (np.linalg.norm(window) ** 2)

    Pxx[1 : n_out - 1] *= 2

    f = np.arange(n_out) / (ceil_pow2(width) * dt)
    
    return n_out, Pxx[: n_out], f