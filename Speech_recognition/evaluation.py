import pandas as pd
import Levenshtein

pred_csv = '~/yourpath/predictions.csv'
true_csv = '~/yourpath/knnw/knnw_en_sub.csv'

df_pred = pd.read_csv(pred_csv)
df_true = pd.read_csv(csv_file, delimiter=";")

def LevenshteinDistance(str1, str2):
    str1 = re.sub(r'[^a-zA-Z0-9 ]', '', str1).lower()
    str2 = re.sub(r'[^a-zA-Z0-9 ]', '', str2).lower()
    distance = Levenshtein.distance(str1, str2)
    return distance

num_items = df_pred.shape[0]
dist = 0.0

for i in range(0, num_items):
    str1 = df_pred['Text'].values[i]
    str2 = df_true['Text'].values[i]
    dist += LevenshteinDistance(str1, str2)

print(dist/num_items)