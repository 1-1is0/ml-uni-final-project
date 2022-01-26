from email.mime import audio
import librosa
import os
import glob
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

file_list = glob.glob("./data/*/*.mp3")


def file_seperate(file_list):
    data = dict()
    for x in file_list:
        if os.path.getsize(x) > 100000:  # 1 K!
            key = x.split('/')[2]
            data.setdefault(key, list()).append(x)
    return data



data = file_seperate(file_list)
print([(k, len(v)) for k,v in data.items()])

class AudioFeature:
    def __init__(self, src_path, fold, label):
        self.src_path = src_path
        self.fold = fold
        self.label = label
        self.y, self.sr = librosa.load(self.src_path, mono=True)
        self.features = None

    def _concat_features(self, feature):
        """
        Whenever a self._extract_xxx() method is called in this class,
        this function concatenates to the self.features feature vector
        """
        self.features = np.hstack(
            [self.features, feature] if self.features is not None else feature
        )

    def _extract_mfcc(self, n_mfcc=25):
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        self._concat_features(mfcc_feature)

    def _extract_spectral_contrast(self, n_bands=3):
        spec_con = librosa.feature.spectral_contrast(
            y=self.y, sr=self.sr, n_bands=n_bands
        )

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        self._concat_features(spec_con_feature)

    def _extract_chroma_stft(self):
        stft = np.abs(librosa.stft(self.y))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=self.sr)
        chroma_mean = chroma_stft.mean(axis=1).T
        chroma_std = chroma_stft.std(axis=1).T
        chroma_feature = np.hstack([chroma_mean, chroma_std])
        self._concat_features(chroma_feature)

    def extract_features(self, *feature_list, save_local=True):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.
        By default the extracted feature and class attributes will be saved in
        a local directory. This can be turned off with save_local=False.
        """
        extract_fn = dict(
            mfcc=self._extract_mfcc,
            spectral=self._extract_spectral_contrast,
            chroma=self._extract_chroma_stft,
        )

        for feature in feature_list:
            extract_fn[feature]()

        if save_local:
            self._save_local()

    def _save_local(self, clean_source=True):
        out_name = self.src_path.split("/")[-1]
        out_name = out_name.replace(".mp3", "")

        filename = f"./data/{self.label}/fold{self.fold}/{out_name}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

        if clean_source:
            self.y = None
    
    def get_pkl_path(self):
        out_name = self.src_path.split("/")[-1]
        out_name = out_name.replace(".mp3", "")

        filename = f"./data/{self.label}/fold{self.fold}/{out_name}.pkl"
        return filename

audio_features = []
key = input("enter a folder") 
value = data.get(key)
print(key)
fold=5
for audio_file in tqdm(value):
    fn = audio_file.split("/")[-1]
    fn = fn.replace(".mp3", "")
    filename = f"./data/{key}/fold{fold}/{fn}.pkl"
    if os.path.isfile(filename):
        print("is file", filename)
    else:
        a = AudioFeature(audio_file, fold, label=key)
        a.extract_features("mfcc", "spectral", "chroma", save_local=True)


print('done')