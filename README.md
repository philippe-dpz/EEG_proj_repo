EEG Data: Motor Imagery Classifier 
==============================

## Project Description

The aim of this project is to develop data visualizations and machine learning models for binary classification using Electroencephalography (EEG) data. Specifically, we are focusing on classifying brain activity related to **Motor Imagery (MI)** for left vs. right-hand imagined movement. This neural pattern is one of the most studied in Cognitive Neuroscience and is detectable through EEG signals.

For a more detailed background on the experiments used to generate the data, check out the official [BCI Competition IV - 2b](http://www.bbci.de/competition/iv/desc_2b.pdf).

In this experiment 2 screening sessions and 1 neurofeedback session per participant was used for the data analysis

## Dataset
The dataset used in this project is **raw EEG data** from the **BCI Competition IV 2b**. The dataset is organized into three main folders:
- `train`: The training data with EEG recordings.
- `test`: The test data that can be used for evaluating models.
- `y_train_only`: The labels corresponding to the training data.

You can download the dataset from Kaggle: [UCSD Neural Data Challenge](https://www.kaggle.com/competitions/ucsd-neural-data-challenge/data).

### Key Features:
- Raw neural data from EEG recordings.
- Data collected from subjects performing motor imagery tasks (left vs. right-hand movement).

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── all_model.ipynb
    │   ├── nf_model_2.ipynb
    │   └── screening_model_2.ipynb
    │
    ├── notebooks          
    │   ├── preprocessing_propre.ipynb
    │   └── vis_propre.ipynb
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   ├── report.pdf
    │   └── references.pdf
    │
    ├── requirements.txt


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
