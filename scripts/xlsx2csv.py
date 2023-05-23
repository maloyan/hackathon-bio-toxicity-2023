import pandas as pd

df = pd.DataFrame(pd.read_excel("data/DeePred-BBB/Table 1.XLSX"))
df.to_csv("data/DeePred-BBB/Table 1.csv", index=False)