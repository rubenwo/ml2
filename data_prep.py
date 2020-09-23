import pandas as pd

df_train = pd.read_json('./data/train.json')

cuisines = {}
uniques = df_train['cuisine'].unique()

for unique in uniques:
    cuisines[unique] = []

for i in range(len(df_train)):
    ingredients = df_train['ingredients'][i]
    cuisines[df_train['cuisine'][i]].extend(ingredients)

for k, v in cuisines.items():
    ingredientcount = {}
    for s in v:
        if s in ingredientcount:
            ingredientcount[s] += 1
        else:
            ingredientcount[s] = 1
    cuisines[k] = ingredientcount

ingr = {}

for k, v in cuisines.items():
    for key in v.keys():
        ingr[key] = []

for k, v in cuisines.items():
    for key in v.keys():
        ingr[key].append(k)

non_unique_ingredients = []
very_unique_ingredients = []

for k, v in ingr.items():
    if len(v) == len(uniques):
        non_unique_ingredients.append(k)

    if len(v) <= 2:
        very_unique_ingredients.append(k)

print(non_unique_ingredients)
print(len(non_unique_ingredients))

print(very_unique_ingredients)
print(len(very_unique_ingredients))
