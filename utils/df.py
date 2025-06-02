import pandas as pd


def df_shape(df):
    """
    Prints the pandas DataFrame shape

    Parameters:
    df (DataFrame): input data frame

    Returns:
    None: no output
    """
    print(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")


def concat_df_cols_old(df, col_prefix):
    """
    Concatenates multiple columns back into a single column.
    
    Args:
        df (pd.DataFrame): The DataFrame with split columns.
        col_prefix (str): The original column name before splitting.

    Returns:
        pd.DataFrame: A DataFrame with concatenated text in a single column.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Identify all column names that belong to the split parts
    split_columns = [col for col in df_copy.columns if col.startswith(f"{col_prefix}_")]

    # Sort columns numerically to ensure correct order
    split_columns.sort(key=lambda x: int(x.split('_')[-1]))

    # Concatenate all parts row-wise
    df_copy[col_prefix] = df_copy[split_columns].fillna("").astype(str).agg("".join, axis=1)

    # Drop the split columns
    df_copy.drop(columns=split_columns, inplace=True)

    return df_copy


def concat_df_cols(df, col_prefix):
    """
    Concatenates multiple columns back into a single column.
    Columns should follow the format - <col_prefix>_<numeric index>
    
    Args:
        df (pd.DataFrame): The DataFrame with split columns.
        col_prefix (str): The original column name before splitting.

    Returns:
        pd.DataFrame: A DataFrame with concatenated text in a single column.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Identify all column names that belong to the split parts
    split_columns = [col for col in df_copy.columns if col.startswith(f"{col_prefix}_")]

    # Sort columns numerically to ensure correct order
    split_columns.sort(key=lambda x: int(x.split('_')[-1]))

    # Concatenate all parts row-wise
    df_copy[col_prefix] = df_copy[split_columns].fillna("").astype(str).agg("".join, axis=1)

    # Drop the split columns
    df_copy.drop(columns=split_columns, inplace=True)

    # Reorder columns to place the concatenated column in the original position of the first split column
    first_split_col_index = df.columns.get_loc(split_columns[0])
    cols = df_copy.columns.tolist()
    cols.insert(first_split_col_index, cols.pop(cols.index(col_prefix)))
    df_copy = df_copy[cols]

    return df_copy


def split_df_col(df, col, max_length=32767):
    """Splits a column with long text into multiple columns."""
    
    def split_long_text(text, max_length):
        """Splits long text into chunks."""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Iterate over each row in the DataFrame
    for idx, row in df_copy.iterrows():
        long_text = row[col]
        # Split the text into smaller parts
        chunks = split_long_text(long_text, max_length)
        # Add the chunks as new columns
        for i, chunk in enumerate(chunks):
            df_copy.at[idx, f'{col}_{i+1}'] = chunk

    return df_copy
