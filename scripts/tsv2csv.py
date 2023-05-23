import pandas as pd
import gzip

# Read in the TSV file
with gzip.open('data/B3DB/B3DB_classification_extended.tsv.gz', 'rt') as f:
    df = pd.read_csv(f, sep='\t')

# Change BBB+ to 1 and BBB- to 0
df["BBB+/BBB-"] = df["BBB+/BBB-"].apply(lambda x: 1 if x == "BBB+" else 0)

# Write out the CSV file
df.to_csv('data/B3DB/B3DB_classification_extended.csv', index=False)


with gzip.open('data/B3DB/B3DB_regression_extended.tsv.gz', 'rt') as f:
    df = pd.read_csv(f, sep='\t')

# Write out the CSV file
df.to_csv('data/B3DB/B3DB_regression_extended.csv', index=False)