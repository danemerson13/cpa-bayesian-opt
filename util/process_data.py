# THIS SCRIPT WILL TAKE IN THE RESULTS FILES AND PARSE THEM INTO THE RESPECTIVE DATA FOLDERS
# DOES NOT NEED TO BE WRITTEN UNTIL AFTER THE FIRST ITERATION OF EXPERIMENTS

import os
import torch
import pandas as pd

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

BATCH_SIZE = 10
ACQUISITION_FUNCTIONS = [
    "Random",
    "LogNParEGO",
    "LogNEHVI",
    "VarLogNEHVI"
]

def load_results(path, filename):
    # This presumes a certain structure of the results file
    # The columns must be named correctly or an error will be raised
    columns = [
        'Index',
        'Glycerol',
        'DMSO',
        'EG',
        '12PD',
        '13PD',
        '3M12PD',
        'Urea',
        'Total Molarity',
        'Experimental average',
        'SD'
    ]
    try:
        df = pd.read_excel(os.path.join(path, filename), usecols = columns, index_col = 0)
        return df
    except ValueError as e:
        print(f"Error loading file {filename}: {e}")
        return None
    
def parse_results(df, iteration):
    for i, acqf in enumerate(ACQUISITION_FUNCTIONS):
        # Convert the correct rows to tensors
        data = torch.tensor(df.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to_numpy()).to(**tkwargs)
        # Save the data to the correct subfolder
        torch.save(data, os.path.join('data', acqf, f'iteration{iteration}.pt'))
    print('All results saved!')

def main():
    # Set the correct iteration and then the results will be saved into the correct subfolders as .pt files
    iteration = 2
    path = 'results'
    filename = f'results_iter{iteration}.xlsx'

    df = load_results(path, filename)
    parse_results(df, iteration)

if __name__ == "__main__":
    main()