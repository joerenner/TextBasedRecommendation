from data_processing import load_data, load_tags, filter_movies_with_no_tags, build_index_dictionary, get_movie_vocab
import random
import math
import os
import numpy as np
import tensorflow as tf

# Parameters
batch_size = 128
embedding_size = 200
window_size = 2
neg_samples = 20
learn_rate = 0.1
num_steps = 500001


# Data Loading
print("loading data...")
# train: {userID (string) => list_of_movies}, train_sets: {userID => set(movies)}, test: {userID => next movie}
train, train_sets, test = load_data(validation=False)
# movie_tags: {movieID (str)=>tagId (str), loads tags (if in user-movie combo is in train set), includes genres as tags
movie_tags, vocab = load_tags(train_sets, min_count=2)
print("done")

train, test = filter_movies_with_no_tags(train, test, movie_tags)
movies_train = list(train.values())
movie_tags_list = list(movie_tags.values())
vocab_size = len(vocab)
num_movies = len(list(movie_tags.keys()))
# avg tags per movie = 17.78249347853091

# build tag => index dictionary and reverse dictionary
tag_dictionary, tag_reversed_dictionary = build_index_dictionary(vocab)
movie_vocab = get_movie_vocab(train)
movie_dictionary, movie_reversed_dictionary = build_index_dictionary(list(movie_vocab))
del train_sets

#del vocab
print("building graph...")
user_index = 0  # where to start generating batch from
tag_index = 0


def generate_batch(batch_size, window_size):
    # window_size = window_size for products
    global user_index
    # src words (batch) and context words (labels)
    batch_words = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels_words = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    batch_movie_words = []
    labels_movie_words = []
    batch_index = 0
    while True:
        num_user_movies = len(movies_train[user_index])
        for i in range(num_user_movies):
            for tag in movie_tags[movies_train[user_index][i]]:
                window_start = max(i - window_size, 0)
                window_end = min(num_user_movies, i + window_size)
                for j in range(window_start, window_end):
                    if i != j:
                        batch_words[batch_index] = tag_dictionary[tag]
                        labels_words[batch_index, 0] = tag_dictionary[random.choice(movie_tags[movies_train[user_index][j]])]

                        batch_movie_words.append(movie_tags[movies_train[user_index][i]])
                        labels_movie_words.append(movie_tags[movies_train[user_index][j]])

                        batch_index += 1
                        if batch_index == batch_size:
                            user_index += 1
                            if user_index == len(movies_train):
                                user_index = 0
                            return batch_words, labels_words, batch_movie_words, labels_movie_words
        user_index += 1
        if user_index == len(movies_train):
            user_index = 0

# main params
lambda_hard = 1

# Model definition
graph = tf.Graph()
with graph.as_default():
    # build ragged movie tags
    a=[]
    b=[]
    for movie,tags in movie_tags.items():
        for tag in tags:
            a.append(movie_dictionary[movie])
            b.append(tag_dictionary[tag])
    movie_tags_ragged = tf.RaggedTensor.from_value_rowids(values=b,value_rowids=a)

    # word input data
    train_word_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_word_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # movie input
    train_movie_inputs = tf.placeholder(tf.int32, shape=[batch_size, None])
    train_movie_labels = tf.placeholder(tf.int32, shape=[batch_size, None])

    # embedding layer
    word_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

    # word embeddings
    word_embed = tf.nn.embedding_lookup(word_embeddings, train_word_inputs)

    # movie embeddings
    # tf.nn.embedding_lookup(word_embeddings, train_movie_inputs) produces shape: (batch_size, None (number of tags for movie), embedding size)
    #movie_input = tf.reduce_mean(tf.nn.embedding_lookup(word_embeddings, train_movie_inputs), 1)
    #movie_label = tf.reduce_mean(tf.nn.embedding_lookup(word_embeddings, train_movie_labels), 1)

    nce_word_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                       stddev=1.0 / math.sqrt(embedding_size)))
    nce_word_biases = tf.Variable(tf.zeros([vocab_size]))

    tag_loss = tf.nn.nce_loss(
            weights=nce_word_weights,
            biases=nce_word_biases,
            labels=train_word_labels,
            inputs=word_embed,
            num_sampled=neg_samples,
            num_classes=vocab_size)

    # entering movie loss

    nce_movie_weights = tf.reduce_mean (tf.ragged.map_flat_values (tf.nn.embedding_lookup, nce_word_weights, movie_tags_ragged), 1)

    movie_embed = tf.reduce_mean (tf.ragged.map_flat_values (tf.nn.embedding_lookup, word_embed, movie_tags_ragged), 1)

    nce_movie_biases = tf.Variable(tf.zeros([num_movies]))

    movie_loss = tf.nn.nce_loss(
            weights=nce_movie_weights,
            biases=nce_movie_biases,
            labels=train_movie_labels,
            inputs=movie_embed,
            num_sampled=neg_samples,
            num_classes=num_movies)

    loss = tf.reduce_mean(tag_loss + lambda_hard * movie_loss)

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keepdims=True))
    normalized_embeddings = word_embeddings / norm

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter("tf_events", session.graph)
    init.run()
    print('graph initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_words, labels_words, batch_movies, labels_movies = generate_batch(batch_size, window_size)
        feed_dict = {train_word_inputs: batch_words, train_word_labels: labels_words, train_movie_inputs: batch_movies, train_movie_labels: labels_movies}

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
    with open("embeddings/PW2V_hard_"+str(embedding_size)+"_"+str(window_size)+"_"+str(neg_samples)+".txt",
              'w+', encoding="utf8") as f:
        for i in range(vocab_size):
            f.write(tag_reversed_dictionary[i] + " ")
            f.write(np.array2string(final_embeddings[i], max_line_width=10000000))
            f.write("\n")

    saver.save(session, os.path.join("tf_events", 'modelPW2V.ckpt'))
    writer.close()
