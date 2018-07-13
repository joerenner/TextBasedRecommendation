from KNearestNeighborsRec import KNearestNeighborsRecModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
import data_processing


class LSIModel(KNearestNeighborsRecModel):

    def __init__(self, movie_tags, vector_size, tfidf=False):
        """
            movie_tags: dict {item_id => tags}
        """
        self.item_vectors = {}
        dictionary = Dictionary()
        for item, tags in movie_tags.items():
            dictionary.add_documents([tags])
        for item, tags in movie_tags.items():
            sparse_vec = dictionary.doc2bow(tags)
            self.item_vectors[item] = sparse_vec
        if tfidf:
            tfidf_model = TfidfModel(self.item_vectors.values())
            for item, vec in self.item_vectors.items():
                tfidf_vec = tfidf_model[vec]
                self.item_vectors[item] = tfidf_vec
        lsi_model = LsiModel(self.item_vectors.values(), num_topics=vector_size)
        for item, vector in self.item_vectors.items():
            self.item_vectors[item] = data_processing.sparse_to_dense(lsi_model[vector], vector_size, norm=True)


print("loading data...")
train, train_sets, test = data_processing.load_data(validation=False)
movie_tags, vocab = data_processing.load_tags(train_sets, min_count=2)
train, test = data_processing.filter_movies_with_no_tags(train, test, movie_tags)
print("performing lsi...")
lsi_model = LSIModel(movie_tags, 100, tfidf=True)
print("getting recs...")
recs = lsi_model.get_recs(train, 10)
hr, ndcg = data_processing.compute_metrics(recs, test)
print(hr)
print(ndcg)
cs_hr, cs_ndcg = data_processing.compute_cold_start_metrics(recs, train, test, window_size=2)
print(cs_hr)
print(cs_ndcg)

