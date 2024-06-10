# News Classifier MLOps Project
This is a MLOps course project. The main purpose of this project is not so much implementing the perfect news classifier as building architecture.

## Project Description:
Problem: Сlassification of news articles. It should help user to be able easily find articles that interest them without having to sift through irrelevant content.

Objective: build a model to classify news articles into categories, which adapts to changes in news topics and language over time.

## High-Level Design:
The system architecture diagram is presented below. 
<img width="468" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/49e059a7-55d4-41b7-861b-3bbc80cac5ea">

It includes 5 main components: User API, ML Server, DB CRUD Server, News Loader, and Monitoring System. The user interacts with the system only via User API, which can call ML Server to classify any news article text. ML Server contains the model, which does the classification, the state of the model is stored in the separate storage. There is a Monitoring System, which permanently does the diagnostics of the model performance. When it underperforms the system triggers a retrain job, which first calls for loading new training data via News Loader and then triggers ML Server to retrain the model. News Loader works according to the ETL principle: first, it triggers the scrappers, which gather the news data from different sources, and then it does the preprocessing of the retrieved data, and finally, it calls DB CRUD Server to store these data inside the database. As expected, DB CRUD Server does all the basic operations with data inside the database. During the retraining ML Server calls DB CRUD Server for the training dataset and saves a new state inside the model storage.
The BERT is selected as a baseline model for the news articles classification. 

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model designed by Google that is excellent in understanding the context of words in a sentence by considering the entire sentence simultaneously in both directions. For text classification, BERT can be fine-tuned on a specific dataset, where it learns to classify texts by leveraging its deep understanding of language nuances and contextual relationships. This fine-tuning process allows BERT to achieve state-of-the-art performance on various text classification tasks.

## Data Requirements:
The basic dataset consists of the data, which consists of 1490 news article texts of 5 classes with labels. The distribution of the classes is the following:
<img width="312" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/6e10f9fb-4364-4aa6-a9b8-6648a888ca9f">
 
The word clouds look like this:
<img width="228" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/0b6fc8b2-1ecc-4680-bd71-2457fe902d3d">
<img width="228" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/7190bcbf-1941-4e5d-84d3-c58c46a451b5">
<img width="228" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/d59ea569-654b-4cf6-a53b-0e7a8cd33b5d">
<img width="228" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/ad18ba9c-dd2a-4d1e-9e8f-0b2716120952">
<img width="228" alt="image" src="https://github.com/dbotuk/news_classifier_mlops/assets/32682272/a15aa810-18f4-4c7f-972b-dc5279e2b822">
   
Data will be updated based on the new articles from the same sources for some selected period. All the newly extracted data will be preprocessed in 5 steps:
1.	Removal of the tags.
2.	Removal of the special characters.
3.	Conversion to the lower case.
4.	Removal of the stop-words.
5.	Lemmatization.
Then the preprocessed article text will be stored with its label in the following form: Id, Text, Label.

## Service Decomposition:
The proposed system will consist of 4 main microservices: ML Server, DB Server, Monitoring System, and News Loader. Additionally, there will be 2-3 microservices, which News Loader will call for scraping data from the news websites. 

The main functionality of the ML Server is obvious – train machine learning, which can classify the news article by its text and provide predictions for the new articles. For this purpose, it calls the DB Server for the training dataset. DB Server is a CRUD server that provides general operations with data in the database. The Monitoring System does the diagnostics of the model performance. When it underperforms the system triggers a retrain job, which first calls for loading new training data via News Loader and then triggers ML Server to retrain the model. News Loader follows a simple ETL approach: first, it calls scrapper to get new data, then it does the transformations and after that, it saves the data into the database via DB Server. All the communication is organized via REST.

## Requirements Specification:
ML Server should have fast inference times for classification tasks. An access to the model and data should be secure during inference and training. 
DB Server should provide all necessary operations with data within database to manage data growth. 

News loader should be flexible to add or modify news sources, and configurable preprocessing steps to adapt new data. It should be able to handle a growing number of news sources and data volume and have efficient resource management for scrapping and preprocessing tasks. 
Monitoring System should provide real-time performance monitoring and alerting. It should be integrated with the rest of the system for seamless operation providing automated triggering or retraining jobs when performance drops below a threshold.

Technologies Stack: Python, Flask, Docker, Kubernetes, Airflow, Monitoring System, PostgreSQL, Minio.

## Evaluation Metrics:
As for any classification problem, standard metrics such as accuracy, precision, recall, and f1 would be pretty good. 

I think if each metric performs with a higher than 95% value, it would be enough to consider the model as well performing.
