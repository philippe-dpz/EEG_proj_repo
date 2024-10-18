"""
#### • Catch22 émulateur

https://r-packages.io/packages/Rcatch22
"""

from .mathesis import *

def CO_Embed2_Dist_tau_d_expfit_meandiff(y : Vector) -> float :
    size : int = len(y)
    tau  : int = co_firstzero(y, size) + 1
    
    if (tau > size / 10) : tau = int(size / 10)

    d = np.sqrt(adjacent(y[: -tau]) ** 2 + adjacent(y[tau :]) ** 2)

    for x in d : 
        if (math.isnan(x)) : return math.nan

    n_bins : int = num_bins_auto(d)

    if (n_bins == 0) : return 0
    
    hist_counts, bin_edges = histcounts_preallocated(d, n_bins)
    
    # normalise to probability
    histCountsNorm = hist_counts / len(d)
    # mean for exponential fit
    mn = np.mean(d)
     
    d_exp_fit_diff = np.zeros(n_bins)

    for i in range(n_bins) :
        expf = math.exp((bin_edges[i] + bin_edges[i + 1]) * -0.5 / mn) / mn

        if (expf < 0) : expf = 0
        
        d_exp_fit_diff[i] = abs(histCountsNorm[i] - expf)
    
    return np.mean(d_exp_fit_diff)

def CO_f1ecac(y : Vector) -> float :
    # Compute autocorrelations
    corrs  = co_autocorrs(y)
    # Threshold to cross
    thresh = 1.0 / math.e

    for i in range(len(y)) :
        if (corrs[i + 1] < thresh) :
            return i + (thresh - corrs[i]) / (corrs[i + 1] - corrs[i])
    
    return len(y) - 1

def CO_FirstMin_ac(y : Vector) -> int :
    size : int = len(y) - 1
    corrs = co_autocorrs(y)

    for i in range(1, size) :
        if ((corrs[i] < corrs[i - 1]) & (corrs[i] < corrs[i + 1])) :
            return i
    
    return size

def CO_HistogramAMI_even_2_5(y : Vector, num_bins : int = 5) -> float :
    tau : int = 2
    # set bin edges
    min_val  = min(y)
    bin_step = (max(y) - min_val + .2) / num_bins

    bins_ext : int = num_bins + 1
    bin_edges = min_val + bin_step * np.arange(bins_ext) - .1
    
    #  count histogram bin contents
    bins1 = histBinAssign(y[: -tau], bin_edges)
    bins2 = histBinAssign(y[tau :], bin_edges)

    # joint
    bins12 = (np.array(bins1) - 1) * bins_ext + bins2

    # fancy solution for joint histogram here
    joint_hist_linear = histcount_edges(bins12, range(1, bins_ext ** 2))
    
    n_bins = range(num_bins)
    # transfer to 2D histogram (no last bin, as in original implementation)
    pij = [[joint_hist_linear[i * bins_ext + j] for j in n_bins] for i in n_bins]
    
    sum_bins = sum([sum(p) for p in pij])
    # normalise
    pij = [p / sum_bins for p in pij]

     # marginals
    pi = np.zeros(num_bins)
    pj = np.zeros(num_bins)

    for i in n_bins :
        for j in n_bins :
            pi[i] += pij[i][j]
            pj[j] += pij[i][j]

    # mutual information
    ami : float = 0

    for i in n_bins :
        for j in n_bins :
            if(pij[i][j] > 0) :
                ami += pij[i][j] * math.log(pij[i][j] / (pj[j] * pi[i]))
    
    return ami

def CO_trev_1_num(y : Vector) -> float :
    return np.mean(adjacent(y) ** 3)

def DN_HistogramMode(y : Vector, n_bins : int = 10) -> float :
    num_max : int = 1
    max_count = 0
    out = 0
    
    num_hists, bin_edges = histcounts_preallocated(y, n_bins)
    
    for i in range(n_bins) :
        if (num_hists[i] > max_count) :
            max_count = num_hists[i]
            num_max  = 1
            out      = (bin_edges[i] + bin_edges[i + 1])
        elif (num_hists[i] == max_count) :
            num_max += 1
            out     += (bin_edges[i] + bin_edges[i + 1])
        
    return out / (num_max << 1)

def DN_Mean(y : Vector) -> float :
    return np.mean(y)

def DN_Outlier_Include_mdrmd(y : Vector, sign : int, inc : float = 1e-2) -> float :
    size   : int  = len(y)
    c_flag : bool = True

    # apply sign, save in new variable
    y_work = y * sign
    tot    = len(np.where(y_work >= 0)[0])

    # apply sign and check constant time series
    for x in y_work :
        if (x != y[0]) :
            c_flag = False

            break

    if (c_flag) : return 0
    
    # find maximum (or minimum, depending on sign)
    max_val = max(y_work)
    
    #  maximum value too small ? return 0
    if (max_val < inc) : return 0
    
    n_thresh = int(max_val / inc) + 1
    
    #  save the median over indices with absolute value > threshold
    ms_dti1 = np.zeros(n_thresh)
    ms_dti3 = np.zeros(n_thresh)
    ms_dti4 = np.zeros(n_thresh)

    lp = range(n_thresh)
    nb = range(size)
    
    k = 100 / tot
    s = 2 / size

    for j in lp :
        it  = j * inc
        # save the indices where y > threshold
        r   = [i + 1 for i in nb if (y_work[i] >= it)]
        #  intervals between high-values
        tmp = adjacent(r)
        
        ms_dti1[j] = np.mean(tmp)
        ms_dti3[j] = k * len(tmp)
        ms_dti4[j] = np.median(r) * s - 1
    
    mj      : int = 0
    trimthr : int = 2
    fbi     : int = n_thresh - 1

    for i in lp :
        if (ms_dti3[i] > trimthr) : mj = i

        k = n_thresh - i - 1

        if (math.isnan(ms_dti1[k])) : fbi = k
    
    return np.median(ms_dti4[: (mj if mj < fbi else fbi) + 1])

def DN_Spread_Std(y : Vector) -> float :
    return np.std(y)

def FC_LocalSimple_mean_tauresrat(y : Vector, train_length : int) -> float :
    size : int = len(y)
    
    res       = fc_local_simple_mean(y, train_length)
    resAC1stZ = co_firstzero(res, size - train_length)
    yAC1stZ   = co_firstzero(y, size)
    
    return resAC1stZ / yAC1stZ

def FC_LocalSimple_mean_stderr(y : Vector, train_length : int) -> float :
    return np.std(fc_local_simple_mean(y, train_length))

def IN_AutoMutualInfoStats_40_gaussian_fmmi(y : Vector, tau : int = 40) -> float :
    size : int = len(y)
    # don't go above half the signal length
    tau = min(tau, math.ceil(size / 2))
    # compute autocorrelations and compute automutual information
    ami = [math.log(1 - autocorr_lag(y, i + 1) ** 2) * -.5 for i in range(tau)]

    # find first minimum of automutual information
    for i in range(1, tau - 1) :
        if((ami[i] < ami[i - 1]) & (ami[i] < ami[i + 1])) : return i

    return tau

def MD_hrv_classic_pnn40(y : Vector, pNNx : int = 40) -> float :
    i = 0

    for x in adjacent(y) :
         if abs(x * 1000) > pNNx : i += 1

    # return len([0 for x in adjacent(y) if abs(x * 1000) > pNNx]) / len(y)
    return i / len(y)

def PD_PeriodicityWang_th0_01(y : Vector, th : float = 1e-2) -> int :
    size  : int = len(y)
    # compute autocorrelations up to 1/3 of the length of the time series
    acmax : int = int(math.ceil(size / 3))
    
    # fit a spline with 3 nodes to the data
    # subtract spline from data to remove trend
    y_sub = np.array(y) - splinefit(y)
    # correlation/ covariance the same, don't care for scaling
    # (cov would be more efficient)
    acf = [cov_mean(y_sub[: -tau], y_sub[tau :]) for tau in range(1, acmax + 1)]

    # find troughts and peaks

    slop_in  = adjacent(acf[: -1])
    slop_out = adjacent(acf[1: ])

    troughs = np.where(((slop_in < 0) & (slop_out > 0)))[0] + 1
    peaks   = np.where(((slop_out < 0) & (slop_in > 0)))[0] + 1

    """
    # s1 = np.zeros(acmax)
    # s2 = np.zeros(acmax)

    for i in range(1, acmax - 1) :
        slopeIn  = acf[i] - acf[i - 1]
        slopeOut = acf[i + 1] - acf[i]

        s1[i - 1] = slopeIn
        s2[i - 1] = slopeOut
        
        if   ((slopeIn < 0) & (slopeOut > 0)) : troughs.append(i)
        elif ((slopeOut < 0) & (slopeIn > 0)) : peaks.append(i)
    """
    
    # search through all peaks for one that meets the conditions:
    # (a) a trough before it
    # (b) difference between peak and trough is at least 0.01
    # (c) peak corresponds to positive correlation
    n_troughs : int = len(troughs)
    
    for i in range(len(peaks)) :
        ip : int = peaks[i]
        # find trough before this peak
        j  : int = -1

        while ((troughs[j + 1] < ip) & (j + 1 < n_troughs - 1)) : j += 1

        # (a) should be implicit
        if (j == -1) : continue
        # (b) different between peak and trough it as least 0.01
        if (acf[ip] - acf[troughs[j]] < th) : continue
        # (c) peak corresponds to positive correlation
        if (acf[ip] < 0) : continue
        
        # use this frequency that first fulfils all conditions.
        return ip
    
    return n_troughs - 1

def SB_MotifThree_quantile_hh(y : Vector, alphabet_size = 3) -> float :
    size : int = len(y)
    inc  : int = size - 1
    # transfer to alphabet
    yt   = sb_coarsegrain(y, 3) #, "quantile"
    abcd = range(alphabet_size) 

    # using selfresizing array for memory efficiency. Time complexity
    # should be comparable due to ammotization.
    Rh = [[j for j in range(size) if (yt[j] == i + 1)] for i in abcd]
    
    # removing last item if it is == max possible idx since later
    # we are taking idx + 1 from yt
    for i in abcd :
        l = len(Rh[i])

        if ((l != 0) & (Rh[i][l - 1] == inc)) : Rh[i] = Rh[i][: l - 1]

    # allocate separately
    out2 = [[len([k for k in Rh[i] if yt[k + 1] == (j + 1)]) / inc for j in abcd]
           for i in abcd]

    return sum([f_entropy(x) for x in out2])

def SB_TransitionMatrix_3ac_sumdiagcov(y : Vector, num_groups : int = 3) -> float :
    size   : int = len(y)
    tau    : int = co_firstzero(y, size)
    n_down : int = int((size - 1) / tau) + 1

    y_down =  [y[i * tau] for i in range(n_down)]
    # transfer to alphabet
    yCG = sb_coarsegrain(y_down, num_groups) # , "quantile"
    T   = np.zeros((num_groups, num_groups))
    
    # more efficient way of doing the below 
    for j in range(n_down - 1) : T[yCG[j] - 1][yCG[j + 1] - 1] += 1
    # print([deviation([T[i][j] / (nDown - 1) for i in grp]) for j in grp])
    
    dev_cols = deviation(T.T) / (n_down - 1)
    cov_tmp  = [np.dot(dev_cols[i], dev_cols[j]) for i in range(num_groups)
                for j in range(i, i + 1)]

    """
    for i in grp :
        for j in range(i, i + 1) : cov_tmp += np.dot(dev_cols[i], dev_cols[j])

    COV = np.zeros((num_groups, num_groups))
    
    for i in range(num_groups) :
        for j in range(i, num_groups) :
            covTemp = np.dot(dev_cols[i], dev_cols[j]) / (num_groups - 1)
            
            COV[i][j] = covTemp
            COV[j][i] = covTemp

    print([COV[i][i] for i in range(num_groups)])
    
    return sum([COV[i][i] for i in range(num_groups)])

    return cov_tmp / (num_groups - 1)
    """
    
    return sum(cov_tmp) / (num_groups - 1)

def SC_FluctAnal_2_50_1_logi_prop_r1(y : Vector, lag : int, how : str) -> float :
    size : int = len(y)
    
    # generate log spaced tau vector
    lin_low  = math.log(5)
    lin_high = math.log(size / 2)
    
    nTauSteps : int = 50
    tauStep = (lin_high - lin_low) / (nTauSteps - 1)

    tau = [int(round(math.exp(lin_low + i * tauStep))) for i in range(nTauSteps)]
    
    # check for uniqueness, use ascending order
    nTau : int = nTauSteps

    for i in range(nTauSteps - 1) :
        while ((tau[i] == tau[i + 1]) & (i < nTau - 1)) :
            tau[i + 1 : nTauSteps - 1] = tau[i + 2 : nTauSteps]
            # for j in range(i + 1, nTauSteps - 1) : tau[j] = tau[j + 1]

            # lost one
            nTau -= 1
    
    # fewer than 12 points -> leave.
    if(nTau < 12) : return 0
    
    sizeCS : int = int(size / lag)
    yCS = np.zeros(sizeCS) # [0 for _ in range(sizeCS)] # malloc(sizeCS * sizeof(double))

    # transform input vector to cumsum
    yCS[0] = y[0]

    # print(y[1 : lag :], [y[(i + 1) * lag] for i in range(sizeCS - 1)])

    for i in range(sizeCS - 1) : yCS[i + 1] = yCS[i] + y[(i + 1) * lag]
    
    # first generate a support for regression (detrending)
    xReg = range(1, tau[nTau - 1] + 1)
    
    # iterate over taus, cut signal, detrend and save amplitude of remaining signal
    F = np.zeros(nTau)

    for i in range(nTau) :
        nBuffer : int = int(sizeCS / tau[i])

        for j in range(nBuffer) :
            p = j * tau[i]

            _, m, b = linreg(xReg, yCS[p :], tau[i])
            
            buffer = [yCS[p + k] - (m * (k + 1) + b) for k in range(tau[i])]
            
            if (how == "rsrangefit") :
                F[i] = (max(max(buffer), tau[i]) - min(min(buffer), tau[i])) ** 2
            elif (how == "dfa") :
                F[i] = sum(np.array(buffer[: tau[i]]) ** 2)
            else :
                return 0.0
        
        if   (how == "rsrangefit") :
            F[i] = math.sqrt(F[i] / nBuffer)
        elif (how == "dfa") :
            F[i] = math.sqrt(F[i] / (nBuffer * tau[i]))

    logtt = np.log(tau[: nTau])
    logFF = np.log(F[: nTau])

    minPoints : int = 6

    sserr = np.zeros(nTau - 2 * minPoints + 1)

    for i in range(minPoints, nTau - minPoints + 1) :
        # this could be done with less variables of course    
        it = i - 1
        
        _ , m1, b1 = linreg(logtt, logFF, i)
        _ , m2, b2 = linreg(logtt[it :], logFF[it :], nTau - it)
        
        # buffer = [logtt[j] * m1 + b1 - logFF[j] for j in range(i)]
        buffer = logtt[: i] * m1 + b1 - logFF[: i]
        
        sserr[i - minPoints] += np.linalg.norm(buffer[: i])

        buffer = [logtt[j + it] * m2 + b2 - logFF[j + it] for j in range(nTau - it)]
        
        sserr[i - minPoints] += np.linalg.norm(buffer[: nTau - it])
    
    minimum = np.min(sserr)

    return ((np.where(sserr == minimum)[0] + 1) / nTau)[0]

def SP_Summaries_welch_rect(y : Vector, what : str, aera : int = 5) -> float :
    size : int = len(y)
    Fs = 1.0    # sampling frequency
    
    # rectangular window for Welch-spectrum / compute Welch-power
    nWelch, S, f = welch(y, ceil_pow2(size), Fs, np.ones(size))
    
    for s in S :
        if (math.isinf(s)) : return 0

    pi_2 = 2 * math.pi
    
    # angualr frequency and spectrum on that
    Sw = S / pi_2

    csS = np.cumsum(Sw)
    
    if (what == "centroid") :        
        csSThres = csS[nWelch - 1] * 0.5

        for i in range(nWelch) :
            if (csS[i] > csSThres) : return f[i] * pi_2
    elif (what == "area_5_1") :
        return sum(S[: int(nWelch // aera)]).real * (f[1] - f[0]) * pi_2
    
    return 0

def DN_HistogramMode_10(y : Vector) -> float :
    return DN_HistogramMode(y, 10)

def DN_HistogramMode_5(y : Vector) -> float :
    return DN_HistogramMode(y, 5)

def DN_OutlierInclude_p_001_mdrmd(y : Vector) -> float :
    return DN_Outlier_Include_mdrmd(y, 1)

def DN_OutlierInclude_n_001_mdrmd(y : Vector) -> float :
    return DN_Outlier_Include_mdrmd(y, -1)

def FC_LocalSimple_mean3_stderr(y : Vector) -> float :
    return FC_LocalSimple_mean_stderr(y, 3)

def FC_LocalSimple_mean1_tauresrat(y : Vector) -> float :
    return FC_LocalSimple_mean_tauresrat(y, 1)

def SB_BinaryStats_diff_longstretch0(y : Vector) -> int :
    return sb_binary_stats_diff_longstretch(adjacent(y) < 0)

def SB_BinaryStats_mean_longstretch1(y : Vector) -> int :
    return sb_binary_stats_diff_longstretch(deviation(y) <= 0)

def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(y : Vector) -> int :
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, 2, "dfa")

def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(y : Vector) -> int :
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, 1, "rsrangefit")

def SP_Summaries_welch_rect_area_5_1(y : Vector) -> float :
    return SP_Summaries_welch_rect(y, "area_5_1")
    
def SP_Summaries_welch_rect_centroid(y : Vector) -> float :
    return SP_Summaries_welch_rect(y, "centroid")

catch_ = [CO_Embed2_Dist_tau_d_expfit_meandiff,     # 0
          CO_f1ecac,
          CO_FirstMin_ac,
          CO_HistogramAMI_even_2_5,
          CO_trev_1_num,
          DN_HistogramMode_10,                      # 5
          DN_HistogramMode_5,
          DN_Mean,
          DN_OutlierInclude_p_001_mdrmd,
          DN_OutlierInclude_n_001_mdrmd,
          DN_Spread_Std,                            # 10
          FC_LocalSimple_mean3_stderr,
          FC_LocalSimple_mean1_tauresrat,
          IN_AutoMutualInfoStats_40_gaussian_fmmi,
          MD_hrv_classic_pnn40,
          PD_PeriodicityWang_th0_01,                # 15
          SB_BinaryStats_diff_longstretch0,
          SB_BinaryStats_mean_longstretch1,
          SB_MotifThree_quantile_hh,
          SB_TransitionMatrix_3ac_sumdiagcov,
          SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,   # 20
          SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
          SP_Summaries_welch_rect_area_5_1,
          SP_Summaries_welch_rect_centroid,]