from data_processing import load_data, get_movie_vocab, build_index_dictionary
import math
import os
import numpy as np
import tensorflow as tf


# Parameters
batch_size = 128
embedding_size = 100
window_size = 3
neg_samples = 20
learn_rate = 1.0

# Data Loading
print("loading data...")
# train: {userID (string) => list_of_movies}, train_sets: {userID => set(movies)}, test: {userID => next movie}
train, train_sets, test = load_data(validation=False)
movies_train = list(train.values())
movie_vocab = get_movie_vocab(train_sets)
vocab_size = len(movie_vocab)
print(vocab_size)
movie_dictionary, movie_reversed_dictionary = build_index_dictionary(list(movie_vocab))

print("building graph...")
user_index = 0  # where to start generating batch from


def generate_batch(batch_size, window_size):
    # window_size = window_size for products
    global user_index
    # src words (batch) and context words (labels)
    batch_words = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels_words = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_index = 0
    while True:
        num_user_movies = len(movies_train[user_index])
        for i in range(num_user_movies):
            window_start = max(i - window_size, 0)
            window_end = min(num_user_movies, i + window_size)
            for j in range(window_start, window_end):
                if i != j:
                    batch_words[batch_index] = movie_dictionary[movies_train[user_index][i]]
                    labels_words[batch_index, 0] = movie_dictionary[movies_train[user_index][j]]
                    batch_index += 1
                    if batch_index == batch_size:
                        user_index += 1
                        if user_index == len(movies_train):
                            user_index = 0
                        return batch_words, labels_words
        user_index += 1
        if user_index == len(movies_train):
            user_index = 0


# Model definition
graph = tf.Graph()
with graph.as_default():
    # word input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    prod_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    prod_embed = tf.nn.embedding_lookup(prod_embeddings, train_inputs)
    nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=prod_embed,
            num_sampled=neg_samples,
            num_classes=vocab_size))

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(prod_embeddings), 1, keepdims=True))
    normalized_embeddings = prod_embeddings / norm

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


num_steps = 500001
with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter("tf_events", session.graph)
    init.run()
    print('graph initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_prod, labels_prod = generate_batch(batch_size, window_size)
        feed_dict = {train_inputs: batch_prod, train_labels: labels_prod}

        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict)
        average_loss += loss_val
        writer.add_summary(summary, step)
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)

    final_embeddings = normalized_embeddings.eval()
    with open("embeddings/P2V_"+str(embedding_size)+"_"+str(window_size)+"_"+str(neg_samples)+"test.txt", 'w+', encoding="utf8") as f:
        for i in range(vocab_size):
            f.write(movie_reversed_dictionary[i] + " ")
            f.write(np.array2string(final_embeddings[i], max_line_width=10000000))
            f.write("\n")

    saver.save(session, os.path.join("tf_events", 'modelP2V.ckpt'))
    writer.close()
