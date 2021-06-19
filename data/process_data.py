import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loading the data 
    Input: 
        messages_filepath = .../data/messages.csv
        categories_filepath = .../data/categories.csv
    Output:
        df: a dataframe from merged messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories datset
    categories = pd.read_csv(categories_filepath)

    # merge datasets using inner
    df = pd.merge(messages, categories, how = 'inner')

    return df

def clean_data(df):
    """
    Cleaning the data
    Input: 
        categories and messages
        drop a column named categories
    Output:
        df: concatenate df and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Apply lambda function and extract a list of new column names for categories. 

    category_names = row.apply(lambda x: x[:-2])

    print(category_names)

    # rename the columns of `categories`
    categories.columns = category_names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('.*(\d{1})', expand = False)

        # convert column from string to numeric
        categories[column] = categories[column].str.extract('(\d+)').astype(int)

    # drop the column names categories    
    df = df.drop(columns = 'categories')
    
    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicate values
    df.drop_duplicates(subset='id', inplace=True)

    # drop duplicated values
    df.drop_duplicates()
    
    # set labels in the 'related' category from 2 to 0
    df.loc[df['related'] > 1,'related'] = 0
    
    # drop 'child_alone' category that all its labels are 0
    df = df.drop(columns = 'child_alone')

    return df

def save_data(df, database_filename):
    """
    Save df as sqlite database (db)
    Input:
        df: cleaned dataset
        database_filename: DisasterResponseMessages.db
    Output:
        A SQLite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False, if_exists='replace')  

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
