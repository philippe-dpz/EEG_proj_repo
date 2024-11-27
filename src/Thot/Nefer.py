### Chargement des différentes librairies
import pandas as pd         # type: ignore
import numpy as np          # type: ignore

import math, time, gc, mne  # type: ignore

from typing_extensions import deprecated # type: ignore

from zipfile import ZipFile
from scipy import signal, stats # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from torch import Tensor # type: ignore

type LComplex = list[complex]
type Clause   = list[str]
type Index    = list[int] | range
type Vector   = list[float] | np.ndarray | Index
type Board    = pd.Series | pd.DataFrame

# %%
class Graphein_DatasLoader(object) :
    runs, target, files = None, None, None
    
    # Correspondance pour la classification
    hands_event = {0: 'Left', 1: 'Right'}
    # Deux enregistrements bipolaires + neutre
    eeg_Chans = ['C3', 'C4', 'Cz']
    # Trois enregistrements musculaires
    ecg_Chans = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']

    def __init__(self, path : str, rep : str, target : bool = True) :        
        if Graphein_DatasLoader.runs == None :
            Graphein_DatasLoader.files = [x[len(f'{rep}/') :] for x in files_in_zip(path, directory = rep)]
            Graphein_DatasLoader.runs  = csv_in_zip(path, directory = rep, files = self.files)

            if target :
                Graphein_DatasLoader.target = csv_in_zip(path, directory = f'y_{rep}_only', files = self.files)
        else :
            print("-- Static Class --")

    # @classmethod  
    # def __getitem__(self, index : int | Index) :
    #     return self.runs[index], self.target[index]

    def __getattribute__(self, name : str) :
        return super(Graphein_DatasLoader, self).__getattribute__(name)

    # def transform(self, threshold : float = 100) :
    #     self.runs_hat = []

    #     for run in self.runs :
    #         self.runs_hat.append(run[abs(run[self.ecg_Chans[0]]) < threshold].reset_index())

    #     return self.runs_hat

# %%   
class Graphein() :
    def __init__(self, datas : list[Board], labels : list[Board] | None, channels : Clause,
                 events : int | Index, chunk_size : int, gap : int, level : bool = True,
                 merge : bool = False, slide : bool = True) :
        runs, self.spots, self.parts = spliting(datas, labels, channels, events, chunk_size, gap,
                                                level = level, merge = merge, slide = slide)
        self.channels = channels
        
        # Regroupement des données en fonction du type de l'évènement et du cannal d'observation
        if merge :
            loop   = range(len(channels))
            n      = len(channels)
            self.X = [[[np.append([], T[j :: n]) for T in runs[i]] for j in loop] for i in events]
        else :
            self.X = [np.concatenate(R, axis = 1) for R in runs]
            self.X = np.concatenate([np.stack(R, axis = 1) for R in self.X])

        self.y = np.concatenate([[i] * (self.X.shape[0] >> 1) for i in events])
        
        del runs
        
        gc.collect()

    def __len__(self) : return len(self.X)
        
    # @overload
    def __getitem__(self, index) : return self.X[index], self.y[index]
    
    def size_str(self) : print('-> X :', *self.X.shape, '| y :', *self.y.shape)
        
    def shuffle(self, transform = None) -> tuple[Vector, Vector] :
        pos = np.random.permutation(range(len(self.X)))

        X = (self.X if transform == None else transform(self.X))

        return X[pos], self.y[pos]
    
    def tensor(self) : return Tensor(self.X), Tensor(self.y)

    # def __getattribute__(self, _) :
    #     return self.trains

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
def single_draw(start : int = 1, end : int = 9, size : int = 2) -> Vector :
    draw = np.random.randint(start, end, size)
    
    draw.sort()
    
    diff = sum(draw[0 : -1 :] == draw[1 ::]) < 1

    return draw if diff else single_draw(start, end, size)

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

    with ZipFile(compressed_files) as zip_file :
        filters = zip_file.infolist() if dir == '' else \
                  [fic for fic in zip_file.infolist() if dir in fic.filename]

        if files != None :
            records = [dir + X for X in files]
            filters = [X for X in filters if (X.filename in records)]

        for fic in filters :
            with zip_file.open(fic.filename) as f :
                df = pd.read_csv(f, encoding_errors = 'ignore')

                res.append(df)

    gc.collect()

    return res

### https://pandas.pydata.org/docs/reference/io.html
def pickle_in_zip(fichier_zip : str, fichier_specifique : str) -> Board :
    with ZipFile(fichier_zip).open(fichier_specifique) as f :
        return pd.read_pickle(f)

### Coefficients du filtre Butterworth pour filtrage passe-bande
def butter_bandpass(lowcut : float, highcut : float, fs : float, order : int | None = 4) \
        -> tuple[Vector, Vector] :
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

###
def titre(txt : str, size : int) -> str :
    n   = len(txt)
    avt = ((size - n) >> 1) - 1

    return f"{'-' * avt} {txt.upper()} {'-' * (size - (avt + n + 1))}"

### %%time
def train_test_init(entrants : list[Board], targets : list[Board], files : Clause,
               methode : int | None = None, reverse : bool = False, test_size : float = .2,
               random_state : int = 42) -> tuple[Clause, Clause, Clause, Clause] :
    n_files = len(files)
    size    = range(n_files)
    files   = np.array(files)
    unic    = len(np.unique([x[2] for x in files]))
    step    = n_files // unic

    match methode :
        case 1 | 2 :
            """
            Le cas 2 est utilisé pour test.
            Les résultats du 'run' 3 ne sont pas utilisés.
            """
            files     = np.array([f for f in files if f not in files[:: -3]])
            size      = range(len(files))
            n_test    = np.array([1, 3]) if methode == 2 else \
                        single_draw(1, unic, math.ceil(.2 * unic))
            test_pos  = [range(i, i + step) for i in (n_test - 1) * step]
            test_pos  = np.append([], test_pos).astype(int)
            train_pos = [i for i in size if i not in test_pos]

            print(*n_test, '\n')
        case 3 :
            """
            Répartition des données d'entrainements et de validation de manière aléatoire (80-20).
            """
            train_pos, test_pos, _, _ = train_test_split(size, size, test_size = test_size,
                                                         random_state = random_state)
        case 4 :
            """
            On utilise les 'runs' 1 et 2 d'un participent tiré au hasard pour les données de validations.
            Et, les résultats du 'run' 3 (sans les données du participant tiré au hasard) pour les données d'entrainements.
            Répartion train : 80 / test : 20 
            """
            n_test    = np.random.randint(1, unic) - 1
            i         = n_test * step
            test_pos  = range(i, i + step)
            train_pos = [i for i in size[:: -3] if i not in test_pos]
            test_pos  = test_pos[: step - 1]

            # print(f"{len(train_pos) / (len(train_pos) + len(test_pos)) :.2%}")
        case _ :
            """
            On utilise les résultats du 'run' 3 pour l'entrainement.
            Et, les résultats des 'runs' 1 et 2 pour les données de validation.
            Il y a moins de données d'ntrainement (1/3) que de validation (2/3)
            """
            train_pos = np.flip(size[:: -3])
            test_pos  = [i for i in size if i not in train_pos]

            # print(f"{len(train_pos) / n_files :.2%}")

    if reverse : train_pos, test_pos = test_pos, train_pos

    # -------------------- Train --------------------
    train_csv   = [entrants[i] for i in train_pos]
    train_label = [targets[i] for i in train_pos]
    # --------------------- Test --------------------
    test_csv    = [entrants[i] for i in test_pos]
    test_label  = [targets[i] for i in test_pos]

    print("Fichiers d'entrainements :\n ", *files[train_pos])
    print()
    print("Fichiers tests :\n ", *files[test_pos])
    print()

    return train_csv, train_label, test_csv, test_label

### %%time
def spliting(datas : list[Board], labels : list[Board] | None, Channels : Clause,
             events : int | Index, chunk_size : int, gap : int, level : bool = True,
             merge : bool = False, slide : bool = True) -> tuple[Index, Index, Index] :
    if type(events) == int : events = range(events)

    temp  = [[] for _ in events]    # Les époques pour tous les cannaux et tous les évènements.
    spots = [[] for _ in events]    # Apparitions des évènements
    parts = []                      #
    # Pour la standardisation du nombre d'échantillon max conservé
    loop  = [x.shape[0] for x in labels]
    ceil  = [min(loop)] * len(datas) if level else loop

    # Extraction des données relavitives à l'apparition des évènements.
    for i in range(len(datas)) :
        input = datas[i]
        types = labels[i].iloc[: ceil[i], -1]                   # labels[i]['EventType'][: ceil[i]]
        sites = np.where(input['EventStart'] == 1)[: ceil[i]]   #

        parts.append(zero_removal(input[Channels[0]], 75))

        for j in events :
            spots[j].append(np.array(*sites)[*np.where(types == j)])

            room = event_epochs(spots[j][-1], chunk_size, gap, slide)

            temp[j].append([full_event(input[c], room, merge) for c in Channels])

    del loop, ceil

    return temp, spots, parts

"""
###
def split_and_merge(datas : list[Board], labels : list[Board] | None, Channels : Clause,
                    events : int | Index, chunk_size : int, gap : int, level : bool = True,
                    merge : bool = False, slide : bool = True) -> tuple[Board, Index, Index] :
    temp, spots, parts = spliting(datas, labels, Channels, events, chunk_size, gap,
                                  level = level, merge = merge, slide = slide)

    # Regroupement des données en fonction du type de l'évènement et du cannal d'observation
    if merge :
        n    = len(Channels)
        temp = [[[np.append([], T[j :: n]) for T in temp[i]] for j in range(n)]
                for i in events]
    else :
        # pool = [[np.stack(x, axis = 0) for x in T] for T in temp]
        # temp = [np.concatenate(R, axis = 1) for R in pool]

        pool = [[[], [], []], [[], [], []]]
        
        [[[[pool[i][j].append(x) for x in A] for j, A in enumerate(T)] for T in temp[i]]
         for i in events]
        
        temp = pool

        # pool  = [[[] for _ in loop] for _ in events]
        # temps = [[[[pool[i][j].append(x) for x in T[j]] for j in loop] for T in runs[i]] for i in events]
        # runs  = pool
    
    eras = [pd.DataFrame({**dict(zip(Channels, [pd.Series(X) for X in temp[i]])), 'EventType': i})
            for i in events]
    
    del pool, temp
    
    gc.collect()

    return eras, spots, parts
"""

###
def torch_split(datas : list[Board], labels : list[Board] | None, Channels : Clause,
                 events : int | Index, chunk_size : int, gap : int, slide : bool = True) -> Tensor :
    temp, _, _ = spliting(datas, labels, Channels, events, chunk_size, gap, merge = False, slide = slide)
    res        = [[np.stack(x, axis = 1) for x in T] for T in temp]
    res        = np.concatenate([np.concatenate(R, axis = 0) for R in res])
    
    del temp
    
    gc.collect()

    return Tensor(np.array(res))

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
def zero_removal(data : Board | Vector, threshold : int | None = 100) -> Vector :
    parts = np.where(abs(data) >= threshold)[0]

    if len(parts) == 0 : return [(0, len(data))]
    
    runs = zero_aggregator(parts)

    return list(zip(np.append(0, runs[1 :: 2] + 1), np.append(runs[0 :: 2], len(data)) - 1))

### Détection des pics positif et négatif dans un signal
def find_peaks_pos(data : Board | Vector, scope : int) -> tuple[Index, Index] :
    pos, _ = signal.find_peaks(data, distance = scope)
    neg, _ = signal.find_peaks(data * -1, distance = scope)
     
    return pos, neg

###
def full_event(data : Board | Vector, tracks : list[Index], flatten : bool = True) -> Vector :
    return np.array(data)[tracks].flatten() if flatten else np.array(data)[tracks]

###
def event_epochs(cuts : Vector | list, period : int, lag : int | None = 0, slide : bool = True) -> Vector :
    if slide : return np.array([range(x + lag, x + lag + period) for x in cuts])

    return np.array([range(x + lag, x + period) for x in cuts])

### 
def normalized(data : Board | Vector) -> Board | Vector :
    return stats.zscore(data)

### 
def filename(data : Clause, start : str = '/', end : str = '.') -> Clause :
    return [x[max(x.rfind(start), 0) : min(max(x.rfind(end), 0), len(x))] for x in data]

###
def moving_average(data : Board | Vector, w : int | None = 3) -> Vector :
    res = data.cumsum() / w

    return np.append(np.zeros(w), res[w :] - res[: -w])

###
def event_slicer(event_type : Board | Vector, cuts : Vector | list, period : int,
                 lag : int | None = 0) -> tuple[Vector, Vector] :
    tout = event_epochs(cuts, period - lag, lag)
    
    return tout[np.where(event_type == 0)], tout[np.where(event_type == 1)]

### 
def simple_thresholding(data : Board | Vector) -> tuple[float, Vector] :
    peak = np.max(abs(data))

    return peak, data / peak

### 
def mne_from_raw(data : Board | Vector, channels : Clause | str, sf : int) -> mne.io.RawArray :
    info = mne.create_info(ch_names = channels, sfreq = sf, ch_types = 'eeg')
    
    return mne.io.RawArray(data.T * 1e-6, info);

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

# %%
d = 511

print((d >> 5) << 5)
print((d // 32) * 32)
# %%