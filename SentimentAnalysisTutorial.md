# Sentiment Analysis using Deep Learning with AMLWorkbench

![Data_Diagram](https://www.usb-antivirus.com/wp-content/uploads/2014/11/tutorial-windwos-10-2-320x202.png)

## Prerequisites

1. Please make sure that you have properly installed Azure ML Workbench by following the [installation guide](https://github.com/Azure/ViennaDocs/blob/master/Documentation/Installation.md).
2. For operationalization, it is best if you have Docker engine installed and running locally. If not, you can use the cluster option but be aware that running an (ACS) Azure Container Service can be expensive.
3. This tutorial assumes that you are running Azure ML Workbench on Windows 10 with Docker engine locally installed. If you are using macOS the instruction is largely the same.

## Tutorial Introduction

Sentiment analysis is a well-known task in the realm of natural language processing. Given a set of texts, the objective is to determine the polarity of that text. The objective of this lab is to use CNTK as the backend for Keras (a model level library, providing high-level building blocks for developing deep learning models) and implement sentiment analysis from movie reviews.

## Use Case Overview

The explosion of data and the proliferation of mobile devices have created lots of opportunities for customers to express their feelings and attitudes about anything and everything at anytime. This opinion or “sentiment” is often generated through social channels in the form of reviews, chats, shares, likes tweets, etc. The sentiment can be invaluable for businesses looking to improve products and services, make more informed decisions, and better promote their brands.

The key to any business success with sentiment data lies in the ability to mine vast stores of unstructured social data for actionable insights. In this tutorial, we will develop deep learning models for performing sentiment analysis of movie reviews using AMLWorkbench

## Data Description

The dataset used to perform sentiment analysis using Deep Learning is a small hand crafted dataset and is located in the Data folder.

The first column contains movie reviews and the second column includes sentiment (0 - negative and 1 - positive). The dataset is merely used for demonstration purposes but typically to get robust sentiment scores, you will need a very large dataset. For example, the [IMDB Movie reviews sentiment classification problem](https://keras.io/datasets/#datasets ) from keras consists of a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). The intention of this lab was to show you how to perform sentiment analysis using Deep Learning with AMLWorkbench.

## Tutorial Structure

The folder structure of the tutorial is arranged as follows:

Code: Contains all the code related to churn prediction using AMLWorkbench  
Data: Contains the dataset used in the tutorial  
Labs: Contains all the hands-on labs  

The order of Hands on Labs to carry out the tutorial is as follows:

1. Data Preparation

   The files related to Data Preparation in the code folder are sampleReviews.dprep, sampleReviews.dprep and sampleReviews.txt

2. Modeling and Evaluation

   The main file related to modeling and evaluation in the code folder is SentimentExtraction.py
   
3. Modeling and Evaluation in Docker

   The main file for this task in the code folder is SentimentExtractionDocker.py

4. Operationalization

   The main files for performing operationalization are the model (sentModel.h5), senti_schema.py, myschema.json and conda_dependencies.yml. They are located in Code/Operaionalization
   
## Conclusion

In conclusion, this tutorial introduces you to using Deep Learning to perform sentiment analysis with the AMLWorkbench. The solution presented in this tutorial is flexible to use CNTK as the backend with Keras. In addition, we also operationalize using HDF5 models.

## Contact

Please feel free to contact Mithun Prasad (miprasad@microsoft.com) with any question or comment.

## Disclaimer

Leave this session as what it is for now. We will update the content once we get more concrete answers from the legal team.
