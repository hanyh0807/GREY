import h5py
import pandas as pd
filename = "similarity_matrix.h5"

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    print(type(f[a_group_key])) 
    data = list(f[a_group_key])

    ds_arr = f[a_group_key][()]  # returns as a numpy array
    
    pd.DataFrame(ds_arr).to_csv("similarity_result.txt")