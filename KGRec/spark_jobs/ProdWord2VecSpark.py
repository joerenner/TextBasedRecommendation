from pyspark import SparkContext, SparkConf
import SparkUtils as utils
from pyspark.mllib.feature import Word2Vec
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='output')
parser.add_argument('-vs', dest='vector_size')
parser.add_argument('-sr', dest='sample_rate')
args = parser.parse_args()

conf = SparkConf().setAppName("pw2v")
sc = SparkContext(conf=conf)
item_words = utils.load_words(sc)
broadcast_item_text = sc.broadcast(item_words)
train = utils.load_dataset(sc, valid_songs=item_words)


def create_tag_sequences(seq, broadcast_var, sample_rate=1):
    seq_length = len(seq)
    new_seqs = []
    item_word_map = broadcast_var.value
    for i in range(seq_length):
        for word in item_word_map[seq[i]]:
            num_prev_words = sample_rate
            num_next_words = sample_rate
            if i != 0:
                num_prev_words = len(seq[i-1])
            if i != seq_length - 1:
                num_next_words = len(seq[i+1])
            for j in range(min(sample_rate, max(num_prev_words, num_next_words))):
                new_seq = []
                if i != 0:
                    new_seq = [random.choice(item_word_map[seq[i - 1]])]
                new_seq.append(word)
                if i != seq_length - 1:
                    new_seq.append(random.choice(item_word_map[seq[i + 1]]))
                new_seqs.append(new_seq)
    return new_seqs


train_seqs = train\
    .flatMap(lambda seq: create_tag_sequences(seq, broadcast_item_text, int(args.sample_rate)))

word2vec = Word2Vec().setVectorSize(args.vector_size).setNumIterations(10).setNumPartitions(10).setWindowSize(2).setMinCount(5)
w2v_model = word2vec.fit(train_seqs)

vectors = w2v_model.getVectors()
utils.save_embeddings(sc, vectors, args.output)

