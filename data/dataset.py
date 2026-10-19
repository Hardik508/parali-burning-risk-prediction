import numpy as np
import pandas as pd

with np.load('real_dataset_matched.npz') as data:
    for filename in data.files:
        arr = data[filename]
        
        # Convert to DataFrame (works best for 1D or 2D arrays)
        df = pd.DataFrame(arr)
        
        # Save to Excel
        df.to_excel(f"{filename}.xlsx", index=False)
