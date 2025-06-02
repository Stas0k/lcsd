import pandas as pd
import os
import re


def get_file_extension(file_name: str):
    """
    Get file extenstion from a file name.
    Example: file_name.csv -> .csv

    Parameters:
    file_name (str): the input file name

    Returns:
    str: The file extension.
    """
    _, file_extension = os.path.splitext(file_name)
    return file_extension.lstrip('.')


def split_excel_to_files(input_path, n_parts, output_dir):
    """
    Splits a large Excel file into N approximately equal parts and saves them to disk.

    Each output file will be named using the format: <original_filename>_part_<i>.xlsx
    All files are saved in the specified output directory.

    Parameters:
    ----------
    input_path (str): Path to the original Excel file to be split.
    n_parts (int): Number of parts to split the file into.

    output_dir (str): Directory where the split Excel files will be saved. Will be created if it doesn't exist.

    Returns:
    -------
    List[str] - A list of file paths to the split Excel files, in order.
    """
    df = pd.read_excel(input_path)
    total_rows = len(df)
    chunk_size = total_rows // n_parts + (total_rows % n_parts > 0)

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Extract base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i in range(n_parts):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        part_df = df.iloc[start_idx:end_idx]
        file_path = os.path.join(output_dir, f"{base_name}_part_{i+1}.xlsx")
        part_df.to_excel(file_path, index=False)
        output_files.append(file_path)
    
    print(f"Saved {n_parts} parts to '{output_dir}'")
    return output_files


def get_file_extension(file_name):
    return os.path.splitext(file_name)[1].lower().lstrip('.')


def extract_part_number(filename):
    """
    Extracts numeric part number from filenames like 'xxx_part_001.xlsx'
    Returns a large number if not found to sort unknowns last
    """
    match = re.search(r'part_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def load_file(file_name, **kwargs):
    """
    Load file to pandas DataFrame. Supported formats: csv, excel.
    If the file doesn't exist, attempts to load from a folder with the same name (no extension) by merging parts.

    Parameters:
    file_name (str): the input file name
    kwargs (dict): keyword arguments

    Returns:
    DataFrame: the data in DataFrame format
    """
    df = None
    try:
        file_format = get_file_extension(file_name)

        if file_format == 'csv':
            df = pd.read_csv(file_name, **kwargs)
        elif file_format.startswith('xls'):
            df = pd.read_excel(file_name, **kwargs)
        else:
            raise ValueError(f"The function doesn't support file format '{file_format}'")

        print(f"Successfully loaded {df.__class__.__name__} from {file_name}")
    
    except FileNotFoundError:
        print(f"File not found: {file_name}. Trying to load from part files...")

        # Get directory and base name
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        parent_dir = os.path.dirname(file_name) or '.'
        folder_path = os.path.join(parent_dir, base_name)

        if os.path.isdir(folder_path):
            part_files = sorted(
                [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')],
                key=lambda f: extract_part_number(f)
            )
            if part_files:
                f_list_str = '\n'.join(pf for pf in part_files)
                print(f"Found {len(part_files)} parts in folder '{folder_path}':\n{f_list_str}")
                df_list = [pd.read_excel(f, **kwargs) for f in part_files]
                df = pd.concat(df_list, ignore_index=True)
                print(f"Successfully reconstructed DataFrame from parts in '{folder_path}'")
            else:
                raise FileNotFoundError(f"No Excel part files found in folder '{folder_path}'")
        else:
            raise FileNotFoundError(f"Neither file '{file_name}' nor folder '{folder_path}' found")
    
    except Exception as e:
        print(f"An error occurred: {e}")

    return df

def save_file(df, file_name):
    """
    Saves a pandas DataFrame to csv or excel.

    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    file_name (str): The name of the Excel file to save the DataFrame to.

    Returns:
    None
    """
    file_format = get_file_extension(file_name)
    try:
        if file_format == 'csv':
             df.to_csv(file_name, index=False)
        elif file_format.startswith('xls'):
            df.to_excel(file_name, index=False)
        else:
            raise ValueError(f"The function doesn't support file format '{file_format}'")
        print(f"{df.__class__.__name__} successfully saved to {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")