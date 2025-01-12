import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import MinMaxScaler
import os
import glob

# ---------------------- CONSTANTES -----------------------
TRAIN_PATH = "../../data/raw/train/"
TRAIN_ONLY_PATH = "../../data/raw/y_train_only/"
MODELS_PATH = "./"
MODELS_TEST_PATH = "./test/"

SAMPLING_FREQ = 250
STICHANNEL = "STI101"
CHANNELS_LIST = ["C3", "Cz", "C4", STICHANNEL]
CH_TYPES = ["eeg", "eeg", "eeg", "stim"]
REFERENCE = [["Cz"]]
MONTAGE = mne.channels.make_standard_montage("standard_1020")
FREQ_BANDS = {
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30],
}

# ---------------------- UTILITAIRES -----------------------
""" Etape de preprocessing d'un fichier de tentatives
    qui aboutit à la créations d'epochs
"""


def file_preprocessing(fileName, lFilter=0, hFilter=30):
    # Lecture de fichier de caractérisques des tentatives et merge avec le fichier de labels
    df_mne = pd.read_csv(TRAIN_PATH + fileName + ".csv")
    df_event_type = pd.read_csv(TRAIN_ONLY_PATH + fileName + ".csv")
    df_event_start = df_mne.loc[(df_mne.EventStart == 1), ["time"]]
    df_event_start = df_event_start.reset_index(drop=True)
    df_event_start_with_type = pd.merge(
        df_event_start, df_event_type, left_index=True, right_index=True
    )
    df_mne = pd.merge(df_mne, df_event_start_with_type, how="left", on=["time", "time"])
    df_mne.drop(
        ["EOG:ch01", "EOG:ch02", "EOG:ch03", "EventStart"], axis=1, inplace=True
    )
    df_mne = df_mne.rename(columns={"EventType": STICHANNEL})

    # Mise à l'échelle des données
    df_mne.C3 = df_mne.C3 / 1000000
    df_mne.Cz = df_mne.Cz / 1000000
    df_mne.C4 = df_mne.C4 / 1000000

    # Labellisation 1 - 2 au lieu de 0 - 1 nécessaire pour MNE
    df_mne.replace({STICHANNEL: 1}, 2, inplace=True)
    df_mne.replace({STICHANNEL: 0}, 1, inplace=True)
    df_mne.fillna({STICHANNEL: 0}, inplace=True)
    data = pd.DataFrame.to_numpy(df_mne[CHANNELS_LIST].transpose(), dtype=np.float64)
    info = mne.create_info(
        ch_names=CHANNELS_LIST,
        sfreq=SAMPLING_FREQ,
        ch_types=CH_TYPES,
    )

    # Création des objets MNE avec intégrations stimuli
    raw = mne.io.RawArray(data, info)
    raw.set_montage(MONTAGE)
    events = mne.find_events(raw, stim_channel=STICHANNEL, consecutive=False)
    mapping = {1: "left", 2: "right"}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )
    raw.set_annotations(annot_from_events)

    # Application d'un filtre passe bande
    raw_filt = raw.filter(l_freq=lFilter, h_freq=hFilter)

    return raw_filt


""" Création des epochs """


def create_epochs(raw, tmin=0, tmax=3.5, reference="average"):
    events = mne.find_events(raw, stim_channel=CHANNELS_LIST, consecutive=False)

    # Création des epochs avec application d'une correction de baseline
    epochs = mne.Epochs(
        raw, events, tmin=tmin, tmax=tmax, baseline=(-0.2, 0), preload=True
    )

    # Application d'une référence aux epochs
    epochs_ref = epochs.set_eeg_reference(ref_channels=reference)
    return epochs_ref


""" Transformation du dataframe pour obtenir une ligne de caractéristiques labellisée par tentative"""


def unstack_df(df):
    dfEvent = df[["id", "epoch", "eventType"]]
    dfEvent.drop_duplicates(inplace=True)
    dfEvent.set_index(["id", "epoch"], inplace=True)
    df = df.astype({"freq": str, "time": str})
    df.set_index(["id", "epoch", "time", "freq"], inplace=True)
    df = df.drop(["C3", "C4", "eventType"], axis=1)
    df = df.unstack()
    df = df.reset_index()
    df.set_index(["id", "time", "epoch"], inplace=True)
    df.columns = pd.MultiIndex.from_frame(
        pd.DataFrame(index=df.columns).reset_index().astype(str)
    )
    df.columns = df.columns.map("_".join)
    df = df.reset_index()
    df.set_index(["id", "epoch", "time"], inplace=True)
    df = df.unstack()
    df.columns = pd.MultiIndex.from_frame(
        pd.DataFrame(index=df.columns).reset_index().astype(str)
    )
    df.columns = df.columns.map("_".join)
    df = df.reset_index()
    dfEvent = dfEvent.reset_index()
    df = pd.merge(df, dfEvent, on=["id", "epoch"])
    df.reset_index(inplace=True)
    return df


""" extraction de caractéristiques 
    Passage en mode fréquentiel sur C3 C4 sur une durée de 1 à 3s sur chaque epoch
    Calcul de la différence C3-C4
    Mise à L'échelle des données
"""


def extract_features(epocks, split, scaler):
    freqs = np.arange(10.5, 12.5, 1)
    dfTrain = []
    dfTest = []
    y_test = []
    idx = 0
    for e in epocks:
        tfr = e.compute_tfr(
            tmin=1,
            tmax=3,
            method="morlet",
            freqs=freqs,
            n_cycles=freqs / 2,
            return_itc=False,
            picks=["C3", "C4"],
            average=False,
        )
        df = tfr.to_data_frame()
        df["id"] = idx
        df.rename(columns={"condition": "eventType"}, inplace=True)
        df.astype({"eventType": int})
        if split:
            if idx > 6:
                dfTest.append(df)
            else:
                dfTrain.append(df)
        else:
            dfTrain.append(df)
        idx += 1

    dfTrain = pd.concat(dfTrain)
    if (scaler == None):
        scaler = MinMaxScaler()
        scaler.fit(dfTrain[["C3","C4"]])
    dfTrain[["C3", "C4"]] = scaler.transform(dfTrain[["C3", "C4"]])    
    dfTrain["C3-C4"] = dfTrain["C3"] - dfTrain["C4"]
    if split:
        dfTest = pd.concat(dfTest)
        dfTest[["C3", "C4"]] = scaler.transform(dfTest[["C3", "C4"]])
        dfTest["C3-C4"] = dfTest["C3"] - dfTest["C4"]

    # Transformation du dataframe pour obtenir une ligne de caractéristiques labellisée par tentative
    dfTrain = unstack_df(dfTrain)
    if split:
        dfTest = unstack_df(dfTest)
        y_test = dfTest["eventType"]
        dfTest.drop(["eventType"], axis=1, inplace=True)    
        dfTest.drop(["id", "index"], axis=1, inplace=True)
        
    y_train = dfTrain["eventType"]
    dfTrain.drop(["eventType"], axis=1, inplace=True)
    dfTrain.drop(["id", "index"], axis=1, inplace=True)

    return dfTrain, dfTest, y_train, y_test, scaler


""" Création d'une animation à partir des données du signal """


def plot_eeg_topomap_animation(
    epochs,
    save_filename_animation,
    start_time=0.00,
    end_time=3.5,
    step_size=0.02,
    frame_rate=10,
):
    conditions = ["1", "2"]
    evoked = {
        c: epochs[c].set_eeg_reference().average()
        for c in conditions
    }
    if save_filename_animation:
        for c in evoked.keys():
            fig, anim = evoked[c].animate_topomap(
                times=np.arange(start_time, end_time, step_size),
                butterfly=True,
                blit=False,
            )
            fig.set_size_inches(2, 2)
            anim.save(MODELS_TEST_PATH + save_filename_animation + "_" + c + ".gif")


def delete_files_in_directory(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, "*"))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
