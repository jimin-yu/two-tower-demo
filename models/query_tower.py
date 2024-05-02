import tensorflow as tf
from tensorflow.keras.layers import StringLookup, Normalization

class QueryTower(tf.keras.Model):

    def __init__(self, user_id_list, emb_dim):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            StringLookup(
                vocabulary=user_id_list,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                # We add an additional embedding to account for unknown tokens.
                len(user_id_list) + 1,
                emb_dim
            )
        ])

        self.normalized_age = Normalization(axis=None)

        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])

    def call(self, inputs):
        concatenated_inputs = tf.concat([
            self.user_embedding(inputs["customer_id"]),
            tf.reshape(self.normalized_age(inputs["age"]), (-1,1)),
            tf.reshape(inputs["month_sin"], (-1,1)),
            tf.reshape(inputs["month_cos"], (-1,1))
        ], axis=1)

        outputs = self.fnn(concatenated_inputs)

        return outputs