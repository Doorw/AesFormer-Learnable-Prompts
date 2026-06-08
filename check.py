import pandas as pd

# pandas 会自动处理引号内的换行符
df = pd.read_csv('data/DPC-Captions/train.csv')
print(f"实际数据条数: {len(df)}")