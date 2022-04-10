# %%
import pandas as pd

df = pd.read_csv('https://drive.google.com/uc?export=download&id=1npVcG_qbzlQeyO0izFVfFKLy5ju1joRG', sep=';')
df.head(20)

df = df.drop(['id_stack', 'views', 'score', 'done'], 1)
#df = df.drop('id_stack', 1)

df.head()

all_tags_list = df['tags'].unique().tolist()
all_tags_list = list(map(lambda x: eval(x), all_tags_list))
all_tags = []
for tags in all_tags_list:
  all_tags.extend(tags)

#print(all_tags)

import collections
counter = collections.Counter(all_tags)
print(counter.most_common(25))

most_common_set = set(map(lambda x: x[0], counter.most_common(25)))
most_common = list(most_common_set)
print(most_common)

new_labels = []
for index in df.index:
  labels = set()
  for tag in eval(df['tags'][index]):
    if tag in most_common_set:
      labels.add(tag)
  
  new_labels.append(list(labels))

df['new_labels'] = new_labels

print(df.tail())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['title'], df['new_labels'], test_size=0.2, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)

from sklearn.preprocessing import MultiLabelBinarizer

# one-hot encoding output labels
encoder = MultiLabelBinarizer()
encoder.fit([most_common])
encoded_y_train = encoder.transform(y_train)
encoded_y_test = encoder.transform(y_test)

print(encoded_y_train)

X_train.sort_indices()
X_test.sort_indices()

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(len(most_common), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, encoded_y_train, epochs=20, validation_data=(X_test, encoded_y_test), batch_size=10)

# %%
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with sess:
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(100, input_dim=X_train.shape[1], activation='relu'))
  model.add(tf.keras.layers.Dense(200, activation='relu'))
  model.add(tf.keras.layers.Dense(200, activation='relu'))
  model.add(tf.keras.layers.Dense(100, activation='relu'))
  model.add(tf.keras.layers.Dense(len(most_common), activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  model.fit(X_train, encoded_y_train, epochs=3, validation_data=(X_test, encoded_y_test), batch_size=10)

  model.save('text_based_model')
# %%
model = tf.keras.models.load_model('text_based_model')

# %%
def guess(sent):
  z = vectorizer.transform([sent])
  z.sort_indices()
  result = model.predict(z)

  top3 = result[0].argpartition(-3)[-3:]

  print(sent)

  for x in top3[::-1]:
    print(encoder.classes_[x], '({:.2f}%)'.format(result[0][x] * 100))
  
  print()


guess('How to write an API in java')
guess('What is a dictionary')
guess('Tuples or lists')
guess('How to give page color and alignment')
guess('How to create a list in javascript')

#print(encoder.classes_)

# %%
