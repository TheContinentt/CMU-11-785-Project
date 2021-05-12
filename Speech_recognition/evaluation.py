import pandas as pd
import Levenshtein

pred_csv = '~/dl_project/predictions.csv'
true_csv = '~/dl_project/knnw/knnw_en_sub.csv'

df_pred = pd.read_csv(pred_csv)
df_true = pd.read_csv(csv_file, delimiter=";")

num_items = df_pred.shape[0]
dist = 0.0
for i in range(0, num_items):
    str1 = df_pred['Text'].values[i]
    str2 = df_true['Text'].values[i]
    dist += LevenshteinDistance(str1, str2)

print(dist/num_items)