import tensorflow as tf
from tensorflow.keras import layers, Model

class Attention(layers.Layer):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention_size = attention_size

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.attention_size), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.attention_size,), initializer='zeros', trainable=True)
        self.v = self.add_weight(shape=(self.attention_size, 1), initializer='random_normal', trainable=True)

    def call(self, inputs):
        q = tf.nn.tanh(tf.linalg.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(tf.linalg.matmul(q, self.v), axis=1)
        output = tf.reduce_sum(inputs * a, axis=1)
        return output

class AttentiveCollaborativeFiltering(Model):
    def __init__(self, num_users, num_items, num_components, embedding_size, attention_size):
        super(AttentiveCollaborativeFiltering, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        self.item_embedding = layers.Embedding(num_items, embedding_size)
        self.component_embedding = layers.Embedding(num_components, embedding_size)
        self.attention = Attention(attention_size)

    def call(self, inputs):
        user_ids, item_ids, component_ids = inputs
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        component_embedding = self.component_embedding(component_ids)

        component_attention = self.attention(component_embedding)
        interaction = tf.multiply(user_embedding, item_embedding + component_attention)
        scores = tf.reduce_sum(interaction, axis=1)
        return scores

# Hyperparameters
num_users = 1000
num_items = 1000
num_components = 10
embedding_size = 64
attention_size = 32

# Instantiate the model
model = AttentiveCollaborativeFiltering(num_users, num_items, num_components, embedding_size, attention_size)

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Train the model with your data
# user_ids, item_ids, and component_ids are arrays with the corresponding ids from the dataset
# labels is an array of binary implicit feedback (1 for interaction, 0 for no interaction)
# model.fit([user_ids, item_ids, component_ids], labels, epochs=10, batch_size=32, validation_split=0.1)




# Data Preparation
import pandas as pd

data = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Data Cleaning
# Remove rows with missing values
data.dropna(subset=['CustomerID', 'StockCode'], inplace=True)

# Convert the CustomerID to integer
data['CustomerID'] = data['CustomerID'].astype(int)

# Generate implicit Feedback 
data['interaction'] = 1

# Map users and items into integers
user_ids = data['CustomerID'].unique()
item_ids = data['StockCode'].unique()

user_to_index = {user: index for index, user in enumerate(user_ids)}
item_to_index = {item: index for index, item in enumerate(item_ids)}

data['user_index'] = data['CustomerID'].map(user_to_index)
data['item_index'] = data['StockCode'].map(item_to_index)

# Extract Required Information
user_indices = data['user_index'].values
item_indices = data['item_index'].values
labels = data['interaction'].values



## MODEL without Components
class AttentiveCollaborativeFilteringWithoutComponents(Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(AttentiveCollaborativeFilteringWithoutComponents, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        self.item_embedding = layers.Embedding(num_items, embedding_size)

    def call(self, inputs):
        user_ids, item_ids = inputs
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        interaction = tf.multiply(user_embedding, item_embedding)
        scores = tf.reduce_sum(interaction, axis=1)
        return scores

# Instantiate the model
model = AttentiveCollaborativeFilteringWithoutComponents(num_users, num_items, embedding_size)

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Train the model with your data
# model.fit([user_indices, item_indices], labels, epochs=10, batch_size=32, validation_split=0.1)


# Predict top K
import numpy as np

def recommend_top_k(model, user_id, user_to_index, item_ids, k, num_items):
    user_index = user_to_index[user_id]
    user_indices = np.full((num_items,), user_index, dtype=int)
    item_indices = np.array(range(num_items))
    
    predictions = model.predict([user_indices, item_indices])
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    top_k_items = [item_ids[i] for i in top_k_indices]
    
    return top_k_items
