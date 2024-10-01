### Chargement des différentes librairies

import pandas as pd
import numpy as np

import time, mne

from typing_extensions import deprecated # type: ignore

from zipfile import ZipFile
from scipy import signal, stats

type Clause = list[str]
type Index  = list[int] | range
type Vector = list[float] | np.ndarray | Index
type Board  = pd.Series | pd.DataFrame

### Décorateur : Temps d'exécution d'une fonction
def temps_execution(function : any) -> any:
    def timer(*args, **kwargs) :
        debut = time.time()
        # ------------------------------------------
        results = function(*args, **kwargs)
        
        print(f"Fonction exécutée en {time.time() - debut} s")
        
        return results
    
    return timer

### 
def samples(n : int, size : float = 1e-1) -> Index :
    res = np.random.choice(n, size = n * 10 if size < 1 else size)

    res.sort()
    
    return res

### 
def random_filter(scope : int | Index, count : int | None = None) -> [int, Index] : # type: ignore
    if type(scope) in [int, float] :
        # pw     = int(abs(scope) * count * .1 ** int(np.log10(count)))
        # scope  = pw if scope < 0 else scope
        res = samples(count, size = scope)
    else :
        res = samples(scope, size = count) if count > 0 else res

    return len(res), res

### 
def files_in_zip(compressed_files : str, directory : str | None = None) -> Clause :
    with ZipFile(compressed_files) as myzip :
        if ((directory == None) | (directory == '')) :
            return [X.filename for X in myzip.infolist()]
        
        dir = f"{directory}/" if directory[-1] != '/' else directory

        return [X.filename for X in myzip.infolist() if dir in X.filename]
    
### https://pandas.pydata.org/docs/reference/io.html
def csv_in_zip(compressed_files : str, directory : str | None = None,
               files : Clause | str | None = None) -> list[Board] | Board :
    res = []
    dir = '' if ((directory == None) | (directory == '')) else f"{directory}/"

    with ZipFile(compressed_files) as myzip :
        filters = myzip.infolist() if dir == '' else \
                  [fic for fic in myzip.infolist() if dir in fic.filename]

        if files != None :
            records = [dir + X for X in files]
            filters = [X for X in filters if (X.filename in records)]

        for fic in filters :
            with myzip.open(fic.filename) as f :
                res.append(pd.read_csv(f, encoding_errors = 'ignore'))

    return res

### https://pandas.pydata.org/docs/reference/io.html
def pickle_in_zip(fichier_zip : str, fichier_specifique : str) -> Board :
    with ZipFile(fichier_zip).open(fichier_specifique) as f :
        return pd.read_pickle(f)

### Coefficients du filtre Butterworth pour filtrage passe-bande
def butter_bandpass(lowcut : float, highcut : float, fs : float, order : int | None = 4) -> tuple[Vector, Vector] :
    return signal.butter(order, 2 * np.array([lowcut, highcut]) / fs, btype = 'band')

### Filtre passe-bande (avec les coefficients b et a issus de la décomposition de Butterworth)
def bandpass_filter(data : Board | Vector, b : Vector, a : Vector) -> Vector :
    return signal.filtfilt(b, a, data)

### Découpage harmonic du signal
def harmonic(data : Vector, bands : dict) -> dict[str: Vector] :
    return {band: bandpass_filter(data, b, a) for band, (b, a) in bands.items()}

### Filtre Notch 
def notch_filter(data : Board | Vector, freq : float, fs : float, window : int, order : int,
                 ripple : float | None = None, filter_type : str | None = None) -> Vector :
    b, a = signal.iirfilter(order, ((2 * freq) + np.array([-window, window])) / fs,
                            rp = ripple, btype = 'bandstop', analog = False, ftype = filter_type)
    
    return signal.lfilter(b, a, data)

### Calcul de l'énergie du signal dans une fenêtre glissante
def signal_energy(data : Board | Vector, window_size : int) -> Vector :
    energy = [np.sum(data[i : i + window_size] ** 2) for i in range(len(data) - window_size)]

    return np.pad(np.array(energy), (0, window_size), 'constant')

### https://fr.wikipedia.org/wiki/Lissage_exponentiel
def simple_exponential_smoothing(data : Board | Vector, alpha : float | None = 1,
                                 s0 : float | None = None) -> Vector :
    res  = data[0] if s0 is None else s0
    beta = 1 - alpha

    for x in data : res.append(alpha * x + beta * res[-1])

    return res

### Sous fonction de run_slicer
def zero_aggregator(data : Board | Vector) -> Vector :
    agg  = [data[0]]                 # On enlève le dernier sample
    last = agg[-1] + 1               # Rajouté ici pour le calcul des différences.
    
    for i in data[1 : ] :
        if i - last > 1 :
            agg.append(last)
            agg.append(i)

        last = i

    return np.append(agg, last)

### Runs are separated by 100 missing values, encoded as the negative maximum values.
### D'après la définition il y a 100 valeurs continue. Mais, on ne tiendra pas compte de cette information.
def zero_removal(data : Board | Vector, step : int | None = 100) -> Vector :
    parts = np.where(abs(data) >= step)[0]

    if len(parts) == 0 : return [range(0, len(data))]
    
    runs = zero_aggregator(parts)
    splt = zip(np.append(0, runs[1 :: 2] + 1), np.append(runs[0 :: 2], len(data)) - 1)

    return [range(a, b) for a, b in splt]

### Détection des pics positif et négatif dans un signal
def find_peaks_pos(data : Board | Vector, scope : int) -> tuple[Index, Index] :
    pos, _ = signal.find_peaks(data, distance = scope)
    neg, _ = signal.find_peaks(data * -1, distance = scope)
     
    return pos, neg

### 
def compare(A : Board | Vector, B : Board | Vector) -> Board :
    return pd.DataFrame(list(A), list(B))
    # return pd.DataFrame({'A' : list(A), 'B' : list(B)})

### 
def full_event(data : Board | Vector, tracks : list[Index], flatten : bool = True) -> Vector :
    return np.array(data)[tracks].flatten() if flatten else np.array(data)[tracks]

### 
def moving_average(data : Board | Vector, w : int | None = 3) -> Vector :
    res = data.cumsum() / w

    return np.append(np.zeros(w), res[w: ] - res[: -w])

### 
def event_epochs(cuts : Vector | list, period : int, lag : int | None = 0) -> Vector :
    return np.array([range(x - lag, x - lag + period) for x in cuts])

### 
def event_slicer(event_type : Board | Vector, cuts : Vector | list, period : int,
                 lag : int | None = 0) -> tuple[Vector, Vector] :
    period -= lag
    tout = event_epochs(cuts, period, lag)
    
    return tout[np.where(event_type == 0)], tout[np.where(event_type == 1)]

### 
def simple_thresholding(data : Board | Vector) -> tuple[float, Vector] :
    peak = np.max(np.abs(data))

    return peak, data / peak

### 
def mne_from_raw(data : Board | Vector, channels : Clause | str, sf : int) -> mne.io.RawArray :
    info = mne.create_info(ch_names = channels, sfreq = sf, ch_types = 'eeg')
    
    return mne.io.RawArray(data.T * 1e-6, info);

### 
def normalized(data : Board | Vector) -> Board | Vector :
    return stats.zscore(data)

### 
def filename(data : Clause, start : str = '/', end : str = '.') -> Clause :
    return [X[max(X.rfind(start), 0) : min(max(X.rfind(end), 0), len(X))] for X in data]

### • Deprecated

@deprecated("Plus utilisé dans le cadre du projet EEG")
def hand_out(data : Board | Vector, events : Index, width : int, channels : Clause,
             hand : int | None = 0, expend : int | None = 0) -> Board :
    xtra   = pd.DataFrame()
    width  += 2 * expend
    signal = [f'S_{i}' for i in range(width)]

    for i in events :
        iexp = i - expend
        span = range(iexp, iexp + width)
        part = [{**{'data_split': iexp},
                 **dict(zip(signal, data.loc[span, c])),
                 **{'hand': hand, f'{c}_dum': 1}} for c in channels]
        xtra = pd.concat([xtra, pd.DataFrame(part)])

    return xtra

@deprecated("Plus utilisé dans le cadre du projet EEG")
### Pour récupérer les évènement relatifs à la survenue d'une action associcée aux mains
def share_out(data : Board | Vector, events : list[Index], size : int,
              canals : Clause, expend : int | None = 0) -> Board :
    hand0 = hand_out(data, events[0], size, canals, 0, expend)
    hand1 = hand_out(data, events[1], size, canals, 1, expend)
    res   = pd.concat([hand0, hand1])

    res.reset_index(drop = True, inplace = True)

    return res

@deprecated("Plus utilisé dans le cadre du projet EEG")
def fancy_df(df : pd.DataFrame, events : list, hands : dict, size : int,
             expend : int = 0) -> tuple[pd.DataFrame, list] : # deprecated
    df_cpy = pd.DataFrame({'C3_4'  : df[['C3', 'C4']].sum(axis = 1), # Somme des signaux 'C3' et 'C4'.
                            # 'EventStart' : df['EventStart'],       # Survenue d'un évènement lié à une des mains.
                           'Hand'  : 0,                              # Activité liée à l'une des mains.
                           'Left'  : 0,                              # Activité liée à la main gauche.
                           'Right' : 0})                             # Activité liée à la main droite.
    evts  = list(zip(np.where(df['EventStart'] == 1)[0], events))
    size += 2 * expend
    # df_cpy['zCore'] = stats.zscore(df['C3_4'])

    for i, j in evts :
        i                        -= expend
        fin                       = range(i, i + size)
        df_cpy.loc[fin, 'Hand']   = np.ones(size)
        df_cpy.loc[fin, hands[j]] = np.ones(size)

    return df_cpy, evts # [i[0] for i in evts]

@deprecated("Plus utilisé dans le cadre du projet EEG")
def left_right_old(df : pd.DataFrame, events : list, size : int, canals, hand : int = 0,
                      expend : int = 0) -> pd.DataFrame :
    lp    = range(len(canals))
    res   = [[] for _ in lp]
    size += expend * 2

    for i in events :
        i   -= expend
        fin = range(i, i + size)

        for k, c in enumerate(canals) :
            res[k] = np.append(res[k], df.loc[fin, c].values)

    return pd.DataFrame({'signal_epoched' : res, 'canal' : canals, 'hand' : hand, 'data_split' : [events for _ in lp]})

# d = 511

# print((d >> 5) << 5)
# print((d // 32) * 32)