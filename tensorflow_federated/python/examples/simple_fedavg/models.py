import functools

import tensorflow as tf


def create_original_fedavg_cnn_model(only_digits=False):
    """The CNN model used in https://arxiv.org/abs/1602.05629.

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.

    Returns:
      An uncompiled `tf.keras.Model`.
    """
    data_format = 'channels_last'
    input_shape = [28, 28, 1]
    init_range = 0.1

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_uniform_initializer(-init_range, init_range),
        bias_initializer=tf.zeros_initializer())

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(2048, activation=tf.nn.relu,
                              kernel_initializer=tf.random_uniform_initializer(-init_range, init_range),
                              bias_initializer=tf.zeros_initializer()),
        tf.keras.layers.Dense(10 if only_digits else 62,
                              kernel_initializer=tf.random_uniform_initializer(-init_range, init_range),
                              bias_initializer=tf.zeros_initializer()),
    ])

    return model


def create_recurrent_model(vocab_size: int,
                           sequence_length: int,
                           mask_zero: bool = True) -> tf.keras.Model:
    """Creates a RNN model using LSTM layers for Shakespeare language models.

    This replicates the model structure in the paper:

    Communication-Efficient Learning of Deep Networks from Decentralized Data
      H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera
      y Arcas. AISTATS 2017.
      https://arxiv.org/abs/1602.05629

    Args:
      vocab_size: the size of the vocabulary, used as a dimension in the input
        embedding.
      sequence_length: the length of input sequences.
      mask_zero: Whether to mask zero tokens in the input.

    Returns:
      An uncompiled `tf.keras.Model`.
    """
    init_range = 0.1

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            input_length=sequence_length,
            output_dim=8,
            mask_zero=mask_zero,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-init_range, maxval=init_range)))
    model.add(
        tf.keras.layers.RNN(
            [tf.keras.layers.LSTMCell(units=256) for _ in range(2)],
            return_sequences=True,
        )
    )
    # lstm_layer_builder = functools.partial(
    #     tf.keras.layers.LSTM,
    #     units=256,
    #     recurrent_activation='sigmoid',
    #     kernel_initializer=tf.keras.initializers.RandomUniform(minval=-init_range, maxval=init_range),
    #     return_sequences=True,
    #     recurrent_dropout=0,
    #     unroll=True,
    #     stateful=False)
    # model.add(lstm_layer_builder())
    # model.add(lstm_layer_builder())
    model.add(tf.keras.layers.Dense(vocab_size,
                                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-init_range, maxval=init_range),
                                    bias_initializer=tf.keras.initializers.Zeros()))  # Note: logits, no softmax.
    return model
