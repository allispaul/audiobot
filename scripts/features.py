# Ripped from the FMA repo, and modified to extract features from 3s windows

import os
import multiprocessing
import ast
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path

AUDIO_DIR = Path("../data/fma_medium")
METADATA_DIR = Path("../data/fma_metadata")
SAVE_PATH = Path("../features/fma_features_3sec.csv")


def get_audio_path(track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(AUDIO_DIR, tid_str[:3], tid_str + '.mp3')


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2, 3])

    filepath = METADATA_DIR / filepath

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            for i in range(size):
                for j in range(10):
                    it = (name, moment, f'{i+1:02d}', f'pos_{j+1:02d}')
                    columns.append(it)

    names = ('feature', 'statistics', 'feature_number', 'song_pos')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def to_chunks(feat):
    """Given a librosa-computed audio feature (calculated over 30s of input),
    of shape [feature_dim, time], reshape it to [feature_dim, 10, time//10].
    Each row [:, i, :] represents the feature calculated over 3s of input.
    """
    return feat[:, :10*(feat.shape[1]//10)].reshape(feat.shape[0], 10, -1)


def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    def feature_stats(name, values):
        # Split 30s values into 3s chunks
        values = np.float32(to_chunks(values))
        # take summary stats along time axis
        features[name, 'mean'] = np.mean(values, axis=-1).ravel()
        features[name, 'std'] = np.std(values, axis=-1).ravel()
        features[name, 'median'] = np.median(values, axis=-1).ravel()
        features[name, 'min'] = np.min(values, axis=-1).ravel()
        features[name, 'max'] = np.max(values, axis=-1).ravel()

        # I removed skewness and kurtosis, which were sometimes undefined when
        # the audio was silent, and are comparatively less useful with the
        # smaller time window.

    try:
        filepath = get_audio_path(tid)
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast
        # check length
        assert 29 < librosa.get_duration(y=x, sr=sr) < 31

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048,
                                               hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        # I'm uncertain about some of these calculations:
        # 1. Should tonnetz be using chroma_cqt or chroma_cens? Does it matter?
        # 2. In general, I got different results from the CQT/STFT compared to
        # the raw audio.

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))

    return features


def main():
    tracks = load('tracks.csv')

    # Only using tracks in fma_medium
    tracks = tracks[tracks["set", "subset"] <= "medium"]

    features = pd.DataFrame(index=tracks.index,
                            columns=columns(), dtype=np.float32)

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))

    # Longest is ~11,000 seconds. Limit processes to avoid memory errors.
    table = ((5000, 1), (3000, 3), (2000, 5), (1000, 10), (0, nb_workers))
    for duration, nb_workers in table:
        print('Working with {} processes.'.format(nb_workers))

        tids = tracks[tracks['track', 'duration'] >= duration].index
        tracks.drop(tids, axis=0, inplace=True)

        pool = multiprocessing.Pool(nb_workers)
        it = pool.imap_unordered(compute_features, tids)

        for i, row in enumerate(tqdm(it, total=len(tids))):
            features.loc[row.name] = row

            if i % 1000 == 0:
                save(features, 10)

    save(features, 10)
    test(features, 10)


def save(features, ndigits):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv(SAVE_PATH, float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = load(SAVE_PATH)
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    main()
