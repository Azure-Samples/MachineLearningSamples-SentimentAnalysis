# Sentiment Analysis using Deep Learning

Run CATelcoCustomerChurnModeling.py in local environment.
```
$ az ml experiment submit -c local SentimentExtraction.py
```

Run iris_sklearn.py in a local Docker container.
```
$ az ml experiment submit -c docker SentimentExtractionDocker.py