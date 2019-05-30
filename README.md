# Yelp Challenge

This repo contains the code for HKUST COMP4332 projects, which is using the data from the [Yelp Challenge](https://www.yelp.com/dataset/challenge). The model for each project is provided under `model.py` or `model.ipynb` on each folder, as well as the training and validation data used. 

`report.pdf` on each project will further discuss the model and features of the data used, as well as further explain the implementation and the final hyperparameters.

## Projects

[Project 1](https://github.com/nwihardjo/yelp_challenge/tree/master/sentiment_analysis): Sentiment Analysis, predicting the rating based on the review provided by the user, mainly the text review is used. The final model uses Bidirectional-GRU with Time-Distributed layers, which able to achieve 70.25% validation accuracy

[Project 2](https://github.com/nwihardjo/yelp_challenge/tree/master/link_prediction): Link Prediction using Deep Walk, predicting the presence of relationship between vertices using DFS-like approach. The final model uses AUC score metrics, and able to achieve 95.87%

[Project 3](https://github.com/nwihardjo/yelp_challenge/tree/master/recommendation_prediction): Recommendation Prediction based on [Wide and Deep Learning](https://arxiv.org/abs/1606.07792) implementation with some feature engineering. RMSE metrics is used, and the final model is able to achieve the value of 1.0293

## Training Environment

Most of the training of the model is done on either [Google Colab](https://colab.research.google.com/) because of their TPU support. However, as running grid search requires significantly longer time to train the model, and Google Colab has its limit on the runtime, [Intel AI Cluster](https://software.intel.com/en-us/articles/getting-started-with-the-intel-ai-devcloud) is used instead.

## Contributor
- [Nathaniel Wihardjo](https://github.com/nwihardjo/)
- [Petra Gabriela](https://github.com/pgabriela/)
- [Mark Uy](https://github.com/mcsuy/)
- [Clyde Ang](https://github.com/ang421/)
- [Hans Krishandi](https://github.com/hskrishandi/)
