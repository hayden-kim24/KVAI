# Last Updated Date: 2024-07-17
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Trying out TensorFlow: Self-Training (Not to be used for actual app)
# Status: Deprecated -- tensorflow not compatible with python -v 3.12

import tensorflow as tf

encoded_sequences = encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = tf.constant(encoded_sequences)