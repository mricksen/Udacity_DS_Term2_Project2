# Udacity Disaster Response Pipeline

## Project Motivation

This is project 2 of the Udacity Data Science Nanodegree Term 2. The project focus is to be able to interpret incoming disaster response data from Figure Eight and classify the response in 35 categories (e.g. water, food, shelter, clothing, etc...).

## File Descriptions
App
  - run.py - Python script to launch the app.
  - Templates
    - master.html - Web page html.
    - go.html - Code to update the highlighted category labels in the webpage after message classification. 
Data
  - disaster_messages.csv - Raw message data from Figure Eight including: translated message, original message, type of message.
  - disaster_categories.csv - Category information per message. 
  - process_data.py - Python script to process the csv data and output as an sqlite3 database.
Models
  - train_classifier.py - Python script to train the classifier model on the processed database.

## Set up instructions 
1. Run the following commands in the project's root directory to set up your database and model.
  - To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`  
  - To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to localhost if running locally
