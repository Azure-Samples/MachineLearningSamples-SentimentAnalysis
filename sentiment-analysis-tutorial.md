# Sentiment Analysis using Deep Learning with Azure Machine Learning

## Link of the Gallery GitHub Repository

Following is the link to the public GitHub repository:

[https://github.com/Azure/MachineLearningSamples-SentimentAnalysis](https://github.com/Azure/MachineLearningSamples-SentimentAnalysis)


## Prerequisites

* An [Azure account](https://azure.microsoft.com/en-us/free/) (free trials are available).

* An installed copy of [Azure Machine Learning Workbench](./overview-what-is-azure-ml.md) following the [quick start installation guide](./quick-start-installation.md) to install the program and create a workspace.

* For operationalization, it is best if you have Docker engine installed and running locally. If not, you can use the cluster option. However, running an Azure Container Service (ACS) can be expensive.

* This Solution assumes that you are running Azure Machine Learning Workbench on Windows 10 with Docker engine locally installed. On a macOS, the instruction is largely the same.

## Introduction

Sentiment analysis is a well-known task in the realm of natural language processing. Given a set of texts, the objective is to determine the polarity of that text. The objective of this solution is to use CNTK as the backend for Keras (a model level library, providing high-level building blocks for developing deep learning models) and implement sentiment analysis from movie reviews.

The solution is located at https://github.com/Azure/MachineLearningSamples-SentimentAnalysis

## Use Case Overview

The explosion of data and the proliferation of mobile devices have created lots of opportunities for customers to express their feelings and attitudes about anything and everything at anytime. This opinion or “sentiment” is often generated through social channels in the form of reviews, chats, shares, likes tweets, etc. The sentiment can be invaluable for businesses looking to improve products and services, make more informed decisions, and better promote their brands.

The success of any business with sentiment data is in the ability to mine vast stores of unstructured social data for actionable insights. In this solution, we will develop deep learning models for performing sentiment analysis of movie reviews using AMLWorkbench

## Data Description

The dataset used to perform sentiment analysis using Deep Learning is a small hand-crafted dataset and is located in the Data folder.

The first column contains movie reviews and the second column includes sentiment (0 - negative and 1 - positive). The dataset is merely used for demonstration purposes but typically to get robust sentiment scores, you need a large dataset. For example, the [IMDB Movie reviews sentiment classification problem](https://keras.io/datasets/#datasets ) from keras consists of a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive or negative). The intention of this lab was to show you how to perform sentiment analysis using Deep Learning with AMLWorkbench.

## Scenario Structure

The folder structure is arranged as follows:

Code: Contains all the code related to churn prediction using AMLWorkbench  
Data: Contains the dataset used in the solution  
Labs: Contains all the hands-on labs  

The order of Hands-on Labs to carry out the solution is as follows:

| Order| File Name | Realted Files |
|--|-----------|------|
| 1 | `DataPreparation.md` | 'Data/sampleReviews.txt' |
| 2 | `ModelingAndEvaluation.md` | 'Code/SentimentExtraction.py.py' |
| 3 | `ModelingAndEvaluationDocker.md` | 'Code/SentimentExtractionDocker.py' |
| 4 | `Operationalization.md` | 'Code/Operaionalization' |

## Conclusion

In conclusion, this solution introduces you to using Deep Learning to perform sentiment analysis with the AMLWorkbench. The solution presented is flexible to use CNTK/Tensorflow as the backend with Keras. In addition, we also operationalize using HDF5 models.

## Contact

Feel free to contact Mithun Prasad (miprasad@microsoft.com) with any question or comment.

## Disclaimer

Leave this session as what it is for now. We will update the content once we get more concrete answers from the legal team.
