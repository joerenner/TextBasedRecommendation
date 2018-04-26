from pyspark import SparkContext, SparkConf
import SparkUtils as utils
from pyspark.mllib.feature import Word2Vec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='output')
parser.add_argument('-vs', dest='vector_size')
parser.add_argument('-ws', dest='window_size')
args = parser.parse_args()

conf = SparkConf().setAppName("p2v")
sc = SparkContext(conf=conf)
train = utils.load_dataset(sc)

word2vec = Word2Vec().setVectorSize(args.vector_size).setNumIterations(300).setWindowSize(args.window_size).setMinCount(1)
w2v_model = word2vec.fit(train)

vectors = w2v_model.getVectors()
utils.save_embeddings(sc, vectors, args.output)


