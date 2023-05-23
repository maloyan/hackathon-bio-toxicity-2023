import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/SkinSensDB/data-20180205.tsv", sep="\t")
df = df.replace(["ND", "NAN", "Below", "NC", "IDR"], np.nan)
df = df.replace(">1000", 1000)
df = df.replace(">2000", 2000)
df = df.replace(">5000", 5000)
df = df.replace("7.7 nmol", 7.7)

# Assuming your DataFrame is named 'df'
# Replace categorical values with dummy variables
# df = pd.get_dummies(df, columns=["Chemical_Name", "CAS No", "PubChem CID", "Canonical SMILES"])

# Encode 'Human_Data' column: N to 0, P to 1, and keep missing values as NaN
df["Human_Data"] = df["Human_Data"].map({"N": 0, "P": 1, pd.NA: pd.NA})

# Convert all columns to numeric, if possible
df = df.apply(pd.to_numeric, errors="ignore")

imputed_df = df[
    [
        "DPRA_PPRA_Cys",
        "DPRA_PPRA_Lys",
        "KeratinoSens_LuSens_EC15",
        "h-CLAT_EC150",
        "h-CLAT_EC200",
        "h-CLAT_CV75",
        "LLNA_EC3",
        "Human_Data",
    ]
]
# # Scale the data
# scaler = StandardScaler()
# scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Impute missing values in 'Human_Data' column using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
imputed_df = pd.DataFrame(
    imputer.fit_transform(
        df[
            [
                "DPRA_PPRA_Cys",
                "DPRA_PPRA_Lys",
                "KeratinoSens_LuSens_EC15",
                "h-CLAT_EC150",
                "h-CLAT_EC200",
                "h-CLAT_CV75",
                "LLNA_EC3",
                "Human_Data",
            ]
        ]
    ),
    columns=[
        "DPRA_PPRA_Cys",
        "DPRA_PPRA_Lys",
        "KeratinoSens_LuSens_EC15",
        "h-CLAT_EC150",
        "h-CLAT_EC200",
        "h-CLAT_CV75",
        "LLNA_EC3",
        "Human_Data",
    ],
)

# Round the imputed 'Human_Data' values to get either 0 (N) or 1 (P)
imputed_df["Human_Data"] = imputed_df["Human_Data"].round().astype(int)
df["Human_Data"] = imputed_df["Human_Data"]

df.to_csv("data/SkinSensDB/data-20180205.csv", index=False)