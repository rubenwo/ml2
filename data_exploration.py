import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_train = pd.read_json('./data/train.json')

# cuisine_count = {}
# uniques = df_train['cuisine'].unique()
#
# for unique in uniques:
#     cuisine_count[str(unique)] = 0
#
# for i in range(len(df_train)):
#     cuisine_count[str(df_train['cuisine'][i])] += 1
#
# for k, v in cuisine_count.items():
#     print("{} occurs {} times in the dataset".format(k, v))

sns.countplot(y='cuisine', data=df_train, palette=sns.color_palette('inferno', 15))
plt.gcf().set_size_inches(15, 10)
plt.title('Cuisine Distribution', size=20)
plt.show()
