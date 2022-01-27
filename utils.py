from cmath import pi
from inspect import getfile
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import glob
import librosa
import pickle
import plotly.graph_objects as go
from tqdm import tqdm
from audio_features import AudioFeature

# Function to Plot Confusion Matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




def plot_table_grid(gs, max_rank):

    table_data = {}
    headers = list(gs.cv_results_['params'][0].keys()) + ['mean_test_score', 'rank']


    for i in np.where(gs.cv_results_['rank_test_score'] <= max_rank)[0]:
        p = gs.cv_results_['params'][i]
        for key, value in p.items():
            table_data.setdefault(key, []).append(value)
        score = gs.cv_results_['mean_test_score'][i]
        table_data.setdefault('mean_test_score', []).append(score)
        rank = gs.cv_results_['rank_test_score'][i]
        table_data.setdefault('rank', []).append(rank)

    fig = go.Figure(data=[go.Table(header=dict(values=headers),
        cells = dict(values=list(table_data.values())))])

    fig.show()


def get_file_dir():

    file_list = glob.glob("./data/*/*.mp3")
    data = dict()
    for x in file_list:
        if os.path.getsize(x) > 100000:  # 100 K!
            key = x.split('/')[2]
            data.setdefault(key, list()).append(x)
    return data


def preprocess():
    data = get_file_dir()
    feature_matrix = []
    labels = []
    for key, value in data.items():
        print(key)
        for audio_file in tqdm(value):
            filename = AudioFeature.get_pkl_path(audio_file, key)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    audio = pickle.load(f)
                    feature_matrix.append(audio.features)
                    labels.append(audio.label)
            else:
                audio = AudioFeature(audio_file, label=key)
                audio.extract_features("mfcc", "spectral", "chroma", save_local=True)
                feature_matrix.append(audio.features)
                labels.append(audio.label)
    
    X = np.vstack(feature_matrix)
    y = np.array(labels)
    with open("./data/X.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open("./data/y.pkl", 'wb') as f:
        pickle.dump(y, f)

def load_data():
    with open("./data/X.pkl", 'rb') as f:
        X = pickle.load(f)
    with open("./data/y.pkl", 'rb') as f:
        y = pickle.load(f)
    return X, y

# X, y = load_data()