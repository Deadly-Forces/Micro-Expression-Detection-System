import sys
import pandas as pd

excel_path = r"D:\Programming\Projects\OpenCV\data\CASME Ⅱ\CASME2-coding-20140508.xlsx"
df = pd.read_excel(excel_path)
print("Columns:", df.columns.tolist())
print(df.head())
