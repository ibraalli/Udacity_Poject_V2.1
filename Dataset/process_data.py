import pandas as pd
from sqlalchemy import create_engine

class DataProcessor:
    def __init__(self):
        pass

    def Read_merge_data(self, path_messages: str, path_categories: str) -> pd.DataFrame:
        """
        Reas messages and categories csv and merge datasets from filepaths.

        Parameters:
            path_messages (str): Filepath to the messages CSV file.
            path_categories (str): Filepath to the categories CSV file.

        Returns:
            pd.DataFrame: Merged DataFrame containing messages and categories.
        """
        # Load datasets
        df_messages = pd.read_csv(path_messages)
        df_categories = pd.read_csv(path_categories)

        # Merge datasets on common id
        df = df_messages.merge(df_categories, how='outer', on='id')
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by splitting categories into separate columns,
        converting category values to binary, and handling duplicates.

        Parameters:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        # Split categories into separate columns
        categories = df['categories'].str.split(';', expand=True)

        # Extract category names
        category_colnames = categories.iloc[0].apply(lambda x: x[:-2])

        # Rename the columns of categories DataFrame
        categories.columns = category_colnames

        # Convert category values to binary
        for column in categories:
            # Set each value to be the last character of the string
            categories[column] = categories[column].astype(str).str[-1]
            # Convert column from string to numeric
            categories[column] = categories[column].astype(int)

        # Replace values of 2 with 1 in 'related' column
        categories['related'] = categories['related'].replace(2, 1)

        # Drop the original categories column from df
        df.drop('categories', axis=1, inplace=True)

        # Concatenate the original DataFrame with the new categories DataFrame
        df = pd.concat([df, categories], axis=1)

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        return df

    def Upload_data(self, df: pd.DataFrame, database_filepath: str) -> None:
        """
        Store DataFrame in a SQLite database.

        Parameters:
            df (pd.DataFrame): DataFrame to be stored.
            database_filepath (str): Filepath for the SQLite database.
        """
        engine = create_engine(f'sqlite:///{database_filepath}')
        df.to_sql('WB_disaster_messages', engine, index=False, if_exists='replace')

    def process_data(self, path_messages: str, path_categories: str, database_filepath: str) -> None:
        """
        Main function to load, clean, and save data to database.

        Parameters:
            path_messages (str): Filepath to the messages CSV file.
            path_categories (str): Filepath to the categories CSV file.
            database_filepath (str): Filepath for the SQLite database.
        """
        # Load data
        print('Reading and Merging data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(path_messages, path_categories))
        df = self.Read_merge_data(path_messages, path_categories)

        # Clean data
        print('Cleaning data...')
        df = self.clean_data(df)

        # Save data
        print('Uploading data...\n    DATABASE: {}'.format(database_filepath))
        self.Upload_data(df, database_filepath)

        print('Cleaned data saved to database!')

# Example usage:
if __name__ == '__main__':
    processor = DataProcessor()
    path_messages = input("Enter filepath for the messages CSV file: ")
    path_categories = input("Enter filepath for the categories CSV file: ")
    database_filepath = input("Enter filepath for the SQLite database: ")
    processor.process_data(path_messages, path_categories, database_filepath)
