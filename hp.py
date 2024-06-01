
import pandas as pd

def look_df(df, num_rows=5, show_types=False):
    """
    This function prints the number of rows and columns in the DataFrame
    and displays the first 5 rows.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to inspect.
    """

    pd.set_option('display.max_columns', None)

    # Print the number of rows and columns
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Print the first num_rows rows
    print(f"\nFirst {num_rows} rows:")
    display(df.head(num_rows))
    
    # Optionally print the column types
    if show_types:
        print("\nColumn types:")
        print(df.dtypes)

def check_dups(df, *cols):
    """
    Check if the combination of specified columns has repeated values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    *cols (str): The column names to check for duplicates.
    
    Returns:
    tuple: A tuple containing two DataFrames:
        - DataFrame of duplicates
        - DataFrame of counts of duplicate combinations
    """
    # Convert cols to a list to ensure correct indexing
    cols = list(cols)
    
    # Group by the specified columns and count the occurrences
    count_df = df.groupby(cols).size().reset_index(name='count')
    
    # Filter for combinations with more than one occurrence
    duplicates_df = count_df[count_df['count'] > 1]
    
    # Filter the original DataFrame for rows that have duplicate combinations
    duplicate_rows_df = df[df.set_index(cols).index.isin(duplicates_df.set_index(cols).index)].reset_index(drop=True)

    look_df(duplicates_df)
    look_df(duplicate_rows_df)
    
    return duplicate_rows_df, duplicates_df

def count_nas(df, columns):
    """
    Calculate the number of NA values in each specified column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for NA values.
    
    Returns:
    pd.DataFrame: DataFrame with columns 'Column' and 'NA_Count' showing the number of NAs in each specified column.
    """
    na_counts = {col: df[col].isna().sum() for col in columns}
    na_counts_df = pd.DataFrame(list(na_counts.items()), columns=['Column', 'NA_Count'])
    return na_counts_df
