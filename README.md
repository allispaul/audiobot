# TEAM AUDIOBOTS: Predicting Music Genres

Our primary goal was to compare the performance of traditional machine learning models with more advanced deep learning models, thereby evaluating the effectiveness of these newer neural network architectures for music genre classification. 

<!-- ## About Team Audiobots
Team members: Aycan Katitas, Dylan Bates, Paul VanKoughnett, Muhammed Cifci, Soheyl Anbouhi, Johann Thiel
-->

# Table of Contents
1. [Introduction](#Introduction)
2. [Data Collection](#Data-Collection)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Modeling Approach](#Modeling-Approach)
5. [Conclusions and Future Directions](#Conclusions-and-Future-Directions)
6. [Description of Repository](#Description-of-Repository)

## Introduction

Historically, songs have been categorized into genres not just for commercial purposes but also to enhance the listening experience and foster cultural exchange through music. With the advent of Music Information Retrieval in the 1990s, researchers began using algorithms to analyze audio files, classifying music based on features like pitch, tempo, and timbre. The abundance of extractable signals from audio files and the rise of deep learning have made genre classification a popular and evolving field among data scientists. In response, we have developed a genre classification system that contributes to these ongoing advancements. 

## Data Collection

Finding a dataset to train genre classifiers was challenging due to licensing issues and low-quality audio files. We used two datasets: [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and [Free Music Archive (FMA)](https://github.com/mdeff/fma). GTZAN is a well-known dataset comprising 1,000 audio tracks, each 30 seconds long, distributed across 10 genres with 100 tracks each. FMA offers two subsets. FMA-small, a balanced dataset with 8,000 tracks of 30 seconds each across 8 genres, and FMA-medium, a larger, unbalanced dataset featuring 25,000 tracks across 16 genres. We initially developed and tested our algorithms on the GTZAN dataset and then moved onto bigger and more complex FMA datasets, using accuracy and F1 score as our evaluation metrics.

## Exploratory Data Analysis


## Modeling Approach

We began by extracting features from audio files and processing them through XGBoost and other classic machine learning algorithms. For deeper insights, we employed sophisticated deep learning techniques in three ways.

First, we used a short-term memory model based on audio features. Second, we generated spectrograms, which are graphical representations of the frequency signals, and analyzed them with Convolutional Neural Networks (CNNs) for image classification. Third, we input raw audio into pretrained transformer models such as DistilHuBERT, Whisper, and Wav2vec, and a version of Google’s WaveNet convolutional architecture.

We trained each of these models on the GTZAN dataset, report accuracy on the unseen test data. Due to the large variation in results, we took the models that performed best and retrained them on FMA Small and FMA Medium, performing model selection and hyperparameter optimization on the validation set.

We report the final accuracy and F1 scores on the test set for FMA Small and FMA Medium below:

**GTZAN**
| Model name/type | Acc. | F1  |
|:--------:|:-------:|:------:|
| LSTM - MFCC | 0.51 | 0.51 |
| WaveNet | 0.67 | 0.57 |
| CNN - MFCC | 0.68 | 0.67 |
| CNN - Spectrogram | 0.85 | 0.85 |
| Wav2vec-base | 0.79 | 0.78 |
| DistilHuBERT | 0.90 | 0.82 |
| **XGBoost** | **0.92** | **0.92** |
| **Whisper Small** | **0.92** | **0.92** |

**FMA Small**
| Model name/type | Acc. | F1  |
|:--------:|:-------:|:------:|
| DistilHuBERT | 0.58 | 0.64 |
| Whisper Small | 0.61 | 0.60 |
| **Whisper Medium** | **0.63** | **0.63** |
| XGBoost | 0.55 | 0.54 |

**FMA Medium**
| Model name/type | Acc. | F1  |
|:--------:|:-------:|:------:|
| Whisper Small | 0.67 | 0.48 |
| **Whisper Medium** | 0.68 | **0.50** |
| XGBoost | **0.72** | 0.35 |

Table: **boldened** values represent the overall best model for that dataset.

## Conclusions and Future Directions


## Description of Repository

The repository is very simple: 
*  All the notebooks we used for visualizing and analyzing the data are in the _EDA_ folder
*  All the notebooks we used for training and evaluating models are in the _models_ folder
*  Finally, we created a [HuggingFace Space](https://huggingface.co/spaces/allispaul/audiobot) where users can upload their audio tracks to classify genres. A copy of the app can be found in the _space_ folder.

A few notebooks of note:

* [Audiobots_Spectrogram_Creation.ipynb](https://github.com/allispaul/audiobot/blob/main/EDA/Audiobots_Spectrogram_Creation.ipynb) - This is a Jupyter Notebook demonstrating how to load the GTZAN data set. It then creates both mel-spectrograms and log-spectrograms of size 224*224 to be input to Convolutional Neural Networks like [InceptionV3](https://huggingface.co/docs/timm/en/models/inception-v3) or [ResNet-34](https://huggingface.co/microsoft/resnet-34).
* [Audiobots_Beyonce.ipynb](https://github.com/allispaul/audiobot/blob/main/models/Audiobots_Beyonce.ipynb) - Extracts features for an album stored locally (like Cowboy Carter). Predicts the genre of each song from the middle 30s of each track using three models pretrained on GTZAN: Whisper Small, DistilHuBERT, and Fleur. Finally, it ouputs csv files containing the logits and probabilities of each genre for further analysis.
