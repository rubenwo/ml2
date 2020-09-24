import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

print("reading training data...")
df_train = pd.read_csv("./data/cooking_train_v2.csv", sep=',')

df_len = len(df_train)
# Split train data in 70-30 train and validation
df_train_t = df_train.iloc[:int(df_len * 0.7)]
df_train_val = df_train.iloc[int(df_len * 0.7):]

print("creating labels...")
labels_strings = df_train['cuisine'].unique()
target_dict = {n: i for i, n in enumerate(labels_strings)}
outputs = df_train['cuisine'].map(target_dict)

features = list(df_train.drop('id', axis=1).drop('cuisine', axis=1).columns)

print("reading test data...")
df_test = pd.read_csv("./data/cooking_test_v2.csv", sep=',')


def compute_accuracy(truth, prediction):
    accuracy = 0
    for truth_i, prediction_i in zip(truth, prediction):
        accuracy += 1 if truth_i == prediction_i else 0

    return accuracy / len(truth)


def compute_rmse(truth, prediction):
    diff = 0

    for truth_i, prediction_i in zip(truth, prediction):
        diff += (1 if truth_i == prediction_i else 0) ** 2

    return (diff / len(truth)) ** (1 / 2)


def compute_auroc(truth, prediction):
    fpr, tpr, threshold = metrics.roc_curve(truth, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


forest = RandomForestClassifier(n_estimators=200, n_jobs=-1)
print("fitting model...")
forest.fit(df_train_t[features], df_train_t['cuisine'])

# Validate model
print("validating model...")
predict_train_t = forest.predict(df_train_t[features])
predict_train_val = forest.predict(df_train_val[features])

validation_result_train_t = compute_rmse(df_train_t['cuisine'], predict_train_t)
validation_result_train_val = compute_rmse(df_train_val['cuisine'], predict_train_val)

print("RMSE Train_T: {}, Accuracy Train_T: {}".format(validation_result_train_t,
                                                      compute_accuracy(df_train_t['cuisine'], predict_train_t)))
print("RMSE Train_VAL: {}, Accuracy Train_VAL: {}".format(validation_result_train_val,
                                                          compute_accuracy(df_train_val['cuisine'], predict_train_val)))
# print("AUC:", compute_auroc(df_train_val['Survived'], predict_train_val))


print('predicting...')
predictions = forest.predict(df_test[features])

print("saving predictions to csv...")
out_csv = pd.DataFrame(columns=['id', 'cuisine'])

for id, prediction in zip(df_test['id'], predictions):
    out_csv = out_csv.append({'id': id, 'cuisine': prediction}, ignore_index=True)

import time

millis = int(round(time.time() * 1000))
# Generate csv
out_csv.to_csv(
    './data/cooking_random_forest-{}-{}.csv'.format(millis,
                                                    compute_accuracy(df_train_val['cuisine'], predict_train_val)),
    index=False)
