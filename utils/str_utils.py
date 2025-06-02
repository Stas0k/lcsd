import numpy as np

def str_to_arr(i_str):
    # convert string to list of numbers
    num_list = list(map(int, i_str.strip('[]').split()))

    # convert to numpy array 
    num_array = np.array(num_list)
    return num_array