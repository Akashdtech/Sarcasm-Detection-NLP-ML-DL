# Sarcasm-Detection-NLP-ML-DL
This project implements a sarcasm detection pipeline using a Kaggle dataset of news headlines. The pipeline preprocesses text data, performs feature engineering, and applies various machine learning and deep learning models to classify headlines as sarcastic or not. The workflow includes:

    Data Preprocessing:
        Text cleaning: removal of non-alphanumeric characters and extra whitespace.
        Stopword removal and lemmatization for meaningful text simplification.

    Feature Engineering:
        Creation of count-based and TF-IDF vectorized representations.
        Combination of features using sparse matrices.

    Class Balancing:
        Upsampling of minority class samples to address dataset imbalance.

    Model Training and Evaluation:
        Machine Learning Models: Linear SVC, Random Forest, and RBF kernel SVC.
        Deep Learning Model: A multi-layer perceptron (MLP) implemented using TensorFlow/Keras.

    Performance Metrics:
        Accuracy and classification reports are generated for each model to evaluate and compare performance.

This repository showcases a comprehensive approach to sarcasm detection using both traditional machine learning techniques and modern deep learning methods.
