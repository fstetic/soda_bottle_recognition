# Classifying soda bottles
This directory contains comparison between deep learning with custom model
and transfer learning on a task of classifying soda bottles using 
https://www.kaggle.com/deadskull7/cola-bottle-identification dataset.\
`clear.py` contains functions for cleaning, preparing and splitting data.\
`data.py` contains functions for loading splitting, and augmenting data.\
`my_model.py` contains custom architecture.\
`pretrained_model.py` contains Tensorflow model pretrained on <insert which
pretrained model you'll use> used for transfer learning approach.\
`metrics.py` contains functions for calculating metrics and 
comparing two approaches.