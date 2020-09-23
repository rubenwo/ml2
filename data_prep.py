import time

import pandas as pd

# Read original data
df_train = pd.read_json('./data/train.json')
df_test = pd.read_json("./data/test.json")

# Create cuisine labels
cuisines = {}
uniques = df_train['cuisine'].unique()

for unique in uniques:
    cuisines[unique] = []

# Add all ingredients in cuisine
for i in range(len(df_train)):
    ingredients = df_train['ingredients'][i]
    cuisines[df_train['cuisine'][i]].extend(ingredients)

# Count the amount of times an ingredient is in a cuisine
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

# Check which ingredients are in all cuisines and in 3 or less cuisines
non_unique_ingredients = []
very_unique_ingredients = []

for k, v in ingr.items():
    if len(v) == len(uniques):
        non_unique_ingredients.append(k)

    if len(v) <= 3:
        very_unique_ingredients.append(k)

# Debug print
print(non_unique_ingredients)
print(len(non_unique_ingredients))
print(very_unique_ingredients)
print(len(very_unique_ingredients))

# Create a cleanable_ingredients list which contains more than 1 word.
cleanable_ingredients = []
for s in very_unique_ingredients:
    if " " in s:
        cleanable_ingredients.append(s)

print(cleanable_ingredients)
print(len(cleanable_ingredients))

# with open("data/cleanable-data-2.txt", 'w', encoding='utf-8') as f:
#     for s in cleanable_ingredients:
#         f.write(s + "\n")
#     f.flush()
#     f.close()

# Read cleaned data dictionary from file
cleaned_data_keys = []
cleaned_data_values = []

with open("data/cleanable-data-2.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        cleaned_data_keys.append(line)
    f.close()

with open("data/cleaned_data.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        s = line.split("=")[1]
        cleaned_data_values.append(s)
    f.close()

cleaned_data_dict = {}
for k, v in zip(cleaned_data_keys, cleaned_data_values):
    cleaned_data_dict[k.strip().lower()] = v.strip().lower()

print(cleaned_data_dict)
print(len(cleaned_data_dict))

print(cleaned_data_dict["Yoplait® Greek 100 blackberry pie yogurt".lower()])

# Create a set all features
original_features = {}
for i in range(len(df_train)):
    ingredients = df_train['ingredients'][i]
    for ingredient in ingredients:
        original_features[str(ingredient).lower()] = True

for i in range(len(df_test)):
    ingredients = df_test['ingredients'][i]
    for ingredient in ingredients:
        original_features[str(ingredient).lower()] = True

print(original_features["Yoplait® Greek 100 blackberry pie yogurt".lower()])
# convert original features to list
original_features = list(original_features.keys())
print(original_features)
print(len(original_features))

# Clean features using the cleaned_data_dictionary
cleaned_features = {}
for k in original_features:
    if k in cleaned_data_dict:
        cleaned_features[(cleaned_data_dict[k])] = True
    else:
        cleaned_features[k] = True

cleaned_features = list(cleaned_features.keys())

# Validate removing worked with a sample ingredient
print("Yoplait® Greek 100 blackberry pie yogurt".lower() in original_features)
print("Yoplait® Greek 100 blackberry pie yogurt".lower() in cleaned_features)

# convert to list for use in new DataFrame
print(cleaned_features)
print(len(cleaned_features))

# Create new DataFrame for one-hot encoded train data
start = time.time()
train_df = pd.DataFrame()
train_df['id'] = df_train['id']
for f in cleaned_features:
    train_df[f] = [0] * len(train_df['id'])

for i in range(len(df_train)):
    ingredients = df_train['ingredients'][i]
    for ingredient in ingredients:
        if ingredient.lower() in cleaned_data_dict:  # check if the ingredient is a cleaned ingredient
            train_df[cleaned_data_dict[ingredient.lower()]][
                i] = 1  # if the ingredient was cleaned, convert it to get the key
        else:
            train_df[ingredient.lower()][i] = 1  # else the column should already exist and we can access it directly

train_df['cuisine'] = df_train['cuisine']

end = time.time()
print(train_df.head())

print(train_df['garlic'][0])  # should be 1
print(train_df['salt'][0])  # should be 0

print("Creating new train data took {} seconds".format(end - start))

# Create new DataFrame for one-hot encoded test data
start = time.time()
test_df = pd.DataFrame()
test_df['id'] = df_test['id']

for f in cleaned_features:
    test_df[f] = [0] * len(test_df['id'])

for i in range(len(df_test)):
    ingredients = df_test['ingredients'][i]
    for ingredient in ingredients:
        if ingredient.lower() in cleaned_data_dict:  # check if the ingredient is a cleaned ingredient
            test_df[cleaned_data_dict[ingredient.lower()]][
                i] = 1  # if the ingredient was cleaned, convert it to get the key
        else:
            test_df[ingredient.lower()][i] = 1  # else the column should already exist and we can access it directly

end = time.time()
print(test_df['milk'][0])  # should be 1
print(test_df['bananas'][0])  # should be 0

print("Creating new test data took {} seconds".format(end - start))
