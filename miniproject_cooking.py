import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v2 as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_core.python.keras.models import load_model

sigmoid = tf.keras.activations.sigmoid
get_custom_objects = tf.keras.utils.get_custom_objects
Activation = tf.keras.layers.Activation


def swish(x, beta=1):
    return x * sigmoid(beta * x)


get_custom_objects().update({'swish': Activation(swish)})

print("reading training data...")
df_train = pd.read_csv("./data/cooking_train_v2.csv", sep=',')

print("creating labels...")
labels_strings = df_train['cuisine'].unique()
target_dict = {n: i for i, n in enumerate(labels_strings)}
outputs = df_train['cuisine'].map(target_dict)

print("reading test data...")
df_test = pd.read_csv("./data/cooking_test_v2.csv", sep=',')

print("defining model...")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('./models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(len(df_train.columns) - 2,)),
    # TODO: probeer eerst grote hidden, gevolgt door kleinere hidden
    # tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    # tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),

    tf.keras.layers.Dense(len(df_train['cuisine'].unique()), activation='softmax')
])

# saved_model = load_model('best_model.h5')
print("compiling model...")
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()],
)

print("fitting model...")
train_history = model.fit(
    x=df_train.drop('id', axis=1).drop('cuisine', axis=1),
    y=outputs,
    epochs=100,
    validation_split=0.3,
    shuffle=True,
    batch_size=32,
    verbose=True,
    callbacks=[es, mc]
)

model = load_model("./models/best_model.h5")

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()

print('predicting...')

preds = model.predict_proba(
    x=df_test.drop('id', axis=1)
)


def find_cuisine(prediction_proba):
    highest = 0
    highest_index = -1
    for i in range(len(prediction_proba)):
        if prediction_proba[i] > highest:
            highest = prediction_proba[i]
            highest_index = i

    for k, v in target_dict.items():
        if v == highest_index:
            return k


predictions = []
for p in preds:
    predictions.append(find_cuisine(p))

out_csv = pd.DataFrame(columns=['id', 'cuisine'])

for id, prediction in zip(df_test['id'], predictions):
    out_csv = out_csv.append({'id': id, 'cuisine': prediction}, ignore_index=True)

import time

millis = int(round(time.time() * 1000))
# Generate csv
out_csv.to_csv('./data/cooking_generated-{}-{}.csv'.format(millis, val_loss[-1]), index=False)
