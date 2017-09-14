# Sentiment analysis using deep learning with Azure Machine Learning Workbench

## Link of the Gallery GitHub Repository
Following is the link to the public GitHub repository where all the codes are hosted:

[https://github.com/Azure/MachineLearningSamples-SentimentAnalysis](https://github.com/Azure/MachineLearningSamples-SentimentAnalysis)


## Prerequisites

* Ensure that you have properly installed Azure Machine Learning Workbench by following the [installation guide](./quick-start-installation.md).

* For operationalization, it is best if you have Docker engine installed and running locally. If not, you can use the cluster option. However, running an (ACS) Azure Container Service can be expensive.

* This Solution assumes that you are running Azure Machine Learning Workbench on Windows 10 with Docker engine locally installed. On a macOS, the instruction is largely the same.

## Tutorial introduction

Sentiment analysis is a well-known task in the realm of natural language processing. Given a set of texts, the objective is to determine the polarity of that text. The objective of this solution is to use CNTK as the backend for Keras (a model level library, providing high-level building blocks for developing deep learning models) and implement sentiment analysis from movie reviews.

The solution is located at https://github.com/Azure/MachineLearningSamples-SentimentAnalysis

## Use case overview

The explosion of data and the proliferation of mobile devices have created lots of opportunities for customers to express their feelings and attitudes about anything and everything at anytime. This opinion or “sentiment” is often generated through social channels in the form of reviews, chats, shares, likes tweets, etc. The sentiment can be invaluable for businesses looking to improve products and services, make more informed decisions, and better promote their brands.

The success of any business with sentiment data is in the ability to mine vast stores of unstructured social data for actionable insights. In this solution, we will develop deep learning models for performing sentiment analysis of movie reviews using AMLWorkbench

## Data description

The dataset used to perform sentiment analysis using Deep Learning is a small hand-crafted dataset and is located in the Data folder.

The first column contains movie reviews and the second column includes sentiment (0 - negative and 1 - positive). The dataset is merely used for demonstration purposes but typically to get robust sentiment scores, you need a large dataset. For example, the [IMDB Movie reviews sentiment classification problem](https://keras.io/datasets/#datasets ) from keras consists of a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive or negative). The intention of this lab was to show you how to perform sentiment analysis using Deep Learning with AMLWorkbench.

## Tutorial structure

The folder structure is arranged as follows:

Code: Contains all the code related to churn prediction using AMLWorkbench  
Data: Contains the dataset used in the solution  
Labs: Contains all the hands-on labs  

The order of Hands-on Labs to carry out the solution is as follows:

1. Data Preparation:
The file related to Data Preparation in the Data folder is sampleReviews.txt
2. Modeling and Evaluation:
The main file related to modeling and evaluation in the code folder is SentimentExtraction.py
3. Modeling and Evaluation in Docker:
The main file for this task in the code folder is SentimentExtractionDocker.py
4. Operationalization:
The main files for performing operationalization are the model (sentModel.h5), senti_schema.py, myschema.json and conda_dependencies.yml. They are located in Code/Operaionalization

## Conclusion

In conclusion, this solution introduces you to using Deep Learning to perform sentiment analysis with the AMLWorkbench. The solution presented is flexible to use CNTK/Tensorflow as the backend with Keras. In addition, we also operationalize using HDF5 models.

## Contact

Feel free to contact Mithun Prasad (miprasad@microsoft.com) with any question or comment.

## Disclaimer

Leave this session as what it is for now. We will update the content once we get more concrete answers from the legal team.
