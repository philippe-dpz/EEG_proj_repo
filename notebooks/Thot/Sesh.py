### Ploting
from .Nefer import *

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

###
def plot_window(data : Board, columns : Clause | str, start : int = 0, size : int | None = None) -> None :
    size : int = (data.shape[1] - start) if (size is None) else size
    end  : int = start + size

    if type(columns) is list :
        for c in columns :
            if '+' in c :
                plt.plot(data[[x.strip() for x in c.split('+')]][start : end].sum(axis = 1), label = c);
            else :
                plt.plot(data[c][start : end], label = c);
    else :
        plt.plot(data[columns][start : end], label = c);

###
def spectrogram_from_EEG(data : Board | Vector, Sampling_rate : int, prior_cut_off : int,
                         later_cut_off : int, view : plt.Axes | None = None) -> None :
    f, t, Sxx = signal.spectrogram(data, fs = Sampling_rate)

    k     = len(f) / f[-1]
    scope = range(int(prior_cut_off * k), int(later_cut_off * k))

    (plt if view == None else view).pcolormesh(t, f[scope], Sxx[scope], shading = 'gouraud')
    
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (sec)")

###
def plot_psd(datas : list[Board], event_type : Board, rate : int,
             Channels : Clause, titled : Clause | None = None) -> None :
    n = len(datas)  # 
    k = 1 << 9      # Taille de la fenètre pour la FFT

    fig, ax = plt.subplots(nrows = n, ncols = 3, figsize = (20, n * 2.5))

    fig.tight_layout(pad = 3.5)

    for i, df in enumerate(datas) :
        view = ax[i, 0]
        lexm = '' if titled == None else titled[i]
        
        view.set_title(lexm)

        for c in Channels :
            view.psd(df[c], NFFT = k, Fs = rate, label = c)

        view.legend(loc = 'upper right');

        for j in range(2) :
            view = ax[i, j + 1]

            view.set_title(f'{lexm} - Main ({j})')

            for c in Channels :
                shape = normalized(event_type.loc[i, f'{c}_{j}'])

                view.psd(shape, NFFT = k, Fs = rate, label = c)

            view.legend(loc = 'upper right');

    plt.show();

###
def subplot_signal(fig : plt.Axes, event0 : Index, event1 : Index, span : int,
                   origin : float, width : int, height : int) -> None :
    mask  = .75

    for i, ev in enumerate([event0, event1]) :
        for x in ev :
            fig.add_patch(Rectangle((x[0], -origin), width = width, height = height,
                                    color = ['teal', 'salmon'][i], alpha = mask))   # '#1B4E88' '#793F22'
    
    fig.hlines(-origin, xmin = 0, xmax = span, colors = '#D03C01')
    fig.hlines( origin, xmin = 0, xmax = span, colors = '#D03C01')

###
def plot_signal(data : Board | Vector, parts : list[tuple[int, int]], event0 : Index, event1 : Index,
                channels : Clause, period : int, lag : int | None = 0, title : str = '') -> None :
    k     = 2 if len(parts) > 1 else 1
    _, ax = plt.subplots(nrows = 3 * k, figsize = (24, 6 * k))

    event0 = event_epochs(event0, period, lag)
    event1 = event_epochs(event1, period, lag)
    pitch  = period << 1 # (event0[0][-1] - event0[0][0]) << 2

    for i, col in enumerate(channels) :
        j     = (i << 1) if len(parts) > 1 else i
        df    = data[col]
        seuil = df.quantile(q = .9975) * 2
        xmax  = len(df)
        view  = ax[j + 0]
        h     = seuil * 2

        peaks_pos, peaks_neg = find_peaks_pos(df, pitch)

        view.plot(df, c = 'gray', label = col, zorder = -1)
        view.plot(peaks_pos, df[peaks_pos], "x", color = 'purple')
        view.plot(peaks_neg, df[peaks_neg], "x", color = 'goldenrod')
        view.legend(loc = 'upper right')
        subplot_signal(view, event0, event1, xmax, seuil, pitch, h)
        
        if len(parts) > 1 :
            view = ax[j + 1]

            for x in parts : view.plot(df[x], zorder = -1)   
            
            subplot_signal(view, event0, event1, xmax, seuil, pitch, h)

    if title != '' : ax[0].set_title(title)

    plt.show();

### Visualiser le signal original et les bandes de fréquences filtrées
def plot_wavelets_z(df : Board, coeffs : dict[str, tuple[float, float]], Channels : Clause,
                    scope = Index, period : int = 0, lag : int | None = 0, titled : str = '') -> None :
    n       = len(Channels)
    _, ax   = plt.subplots(nrows = 2, ncols = n, figsize = (n * 6.5, 3.6), sharex = 0)

    # print(scope)
    # print(event_epochs(scope, period, lag))

    scope   = list(event_epochs(scope, period, lag))
    x_ticks = df.loc[scope, Channels[0]].index
    titled += " - " if titled != '' else ''

    for i, col in enumerate(Channels) :
        view    = ax[0, i]
        data    = normalized(df.loc[scope, col])
        signals = {band : bandpass_filter(data, b, a) for band, (b, a) in coeffs.items()}

        view.plot(data, label = 'Raw signal')
        view.plot(pd.Series(signals[[*signals][0]], x_ticks), '--', c = 'maroon', label = 'Porteuse') # darkviolet indigo firebrick tomato darkturquoise
        view.set_title(titled + col)
        view.legend(loc = 'upper right')
        
        view = ax[1, i]

        for (band, signal) in reversed(signals.items()) :
            view.plot(pd.Series(signal, x_ticks), label = f'{band}', c = np.random.rand(1, 3)[0])

        view.legend(loc = 'upper right')

    plt.tight_layout()
    plt.show();

### Visualiser le signal original et les bandes de fréquences filtrées
def plot_wavelets(data : Board, coeffs : dict[str, tuple[float, float]], Channels : Clause,
                  scope : int = -1, titled : Clause | None = None) -> None :
    count  = len(data.index)
    pw     = int(abs(scope) * count * .1 ** int(math.log10(count)))
    scope  = pw if scope < 0 else scope
    sample = np.random.default_rng().integers(count, size = scope)
    
    n_tick = len(data.iloc[0, 0])
    n_titl = count // len(titled)
    n      = len(Channels)
    _, ax  = plt.subplots(nrows = 2 * scope, ncols = n,
                          figsize = (6.5 * n, 3.6 * scope), sharex = 0)

    sample.sort()

    for i, k in enumerate(sample) :
        n       = k // n_titl
        x_ticks = range(n, n + n_tick)
        lexem   = f"{titled[n]} - " if titled != None else ''
        
        for j, col in enumerate(Channels) :
            view    = ax[2 * i + 0, j]
            bw      = normalized(data[col][k])
            signals = {band : bandpass_filter(bw, b, a) for band, (b, a) in coeffs.items()}
            
            view.plot(pd.Series(bw, x_ticks), label = 'Raw signal')
            view.plot(pd.Series(signals[[*signals][0]], x_ticks), '--', c = 'maroon', label = 'Porteuse') # darkviolet indigo firebrick tomato darkturquoise
            view.set_title(lexem + col)
            view.legend(loc = 'upper right')
        
            view = ax[2 * i + 1, j]

            for (band, signal) in reversed(signals.items()) :
                view.plot(pd.Series(signal, x_ticks), label = f'{band}', c = np.random.rand(1, 3)[0])

            view.legend(loc = 'upper right')

    plt.tight_layout()
    plt.show();

###
# def plot_wavelets(df : Board, coeffs : dict[str, tuple[float, float]],
#                   scope = Index, label : str = '') -> None :
#     # n       = len(Channels)
#     _, ax   = plt.subplots(nrows = 2, ncols = n, figsize = (n * 7, 3.6), sharex = 0)
#     x_ticks = df.loc[scope, Channels[0]].index
#     label   += " - " if label != '' else ''

#     for i, col in enumerate(Channels) :
#         view    = ax[0, i]
#         data    = normalized(df.loc[scope, col])
#         signals = {band : bandpass_filter(data, b, a) for band, (b, a) in coeffs.items()}

#         view.plot(data, label = 'Raw signal')
#         view.plot(pd.Series(signals[[*signals][0]], x_ticks), '--', c = 'maroon', label = 'Porteuse') # darkviolet indigo firebrick tomato darkturquoise
#         view.set_title(label + col)
#         view.legend(loc = 'upper right')
        
#         view = ax[1, i]

#         for (band, signal) in reversed(signals.items()) :
#             view.plot(pd.Series(signal, x_ticks), label = f'{band}', c = np.random.rand(1, 3)[0])

#         view.legend(loc = 'upper right')

#     plt.tight_layout()
#     plt.show();