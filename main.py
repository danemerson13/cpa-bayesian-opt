from src import mobo
import pandas as pd

import warnings
from linear_operator.utils.warnings import NumericalWarning
warnings.filterwarnings("ignore", category=NumericalWarning)

ACQUISITION_FUNCTIONS = [
    "Random",
    "LogNParEGO",
    "LogNEHVI",
    "VarLogNEHVI"
]

COLUMNS = [
    "Glycerol", "DMSO", "EG", "12PD", "13PD", "3M12PD", 
    "Urea", "Total Molarity", "Predicted Viability"
]

def main():
    all_candidates = []

    for acq_name in ACQUISITION_FUNCTIONS:
        candidates = mobo.run(acq_name)  # Assuming this returns a list of 10 candidates
        for i, candidate in enumerate(candidates, start=1):
            index_label = f"{acq_name} {i}"
            all_candidates.append([index_label] + list(candidate))

    # Create a DataFrame with the specified columns
    df = pd.DataFrame(all_candidates, columns=["Index"] + COLUMNS)
    df.set_index("Index", inplace=True)

    # Write the DataFrame to an Excel file
    df.to_excel("candidates.xlsx")

    print("All Done!")

if __name__ == "__main__":
    main()