import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages_df, categories_df, on='id')
    
    return df

def clean_data(df):
    categories = df.categories.str.split(";", expand=True)
    
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    categories.related.replace(2, 1, inplace=True)
    
    df.drop(['categories','original'], axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1, sort=False)
    
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Uncomment below code to create new table if make any updates and need to
    # reprocess data
    '''conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    cur.execute('DROP TABLE disaster_messages;')'''
    
    df.to_sql('disaster_messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
