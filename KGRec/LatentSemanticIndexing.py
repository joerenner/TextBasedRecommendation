from KNearestNeighborsRec import KNearestNeighborsRecModel
from gensim.models import LsiModel
import numpy as np
from KGRec import Utils


class LatentSemanticIndexingModel(KNearestNeighborsRecModel):
    def __init__(self, vector_size=100):
        """ initializes LSI recommendation model
            Parameters
            ----------
                vector_size : int, size of low rank vectors
            Attributes
            ----------
                vector_size
                k : int, number of recommendations to generate per user
                vectors : dict of (song_id => lsi vectors)
                user_vectors : dict of (user_id => lsi vectors)
                tag_dict : gensim Dictionary of (tag => index)
                tfidf_model : gensim TfIdf model (for transforming songs not in original training)
                lsi_model : gensim lsi model (for transforming songs not in original training) 
        """
        self.vector_size = vector_size
        self.item_vectors = {}
        self.user_vectors = {}
        self.tag_dict = {}
        self.tfidf_model = {}
        self.lsi_model = {}

    def compute_vectors(self):
        """ build sparse vectors, computes tfidf matrix, performs svd, saves low rank vectors
            saves tag_dict, tfidf_model, lsi_model, and item_vectors to object
        """
        id_vec_mapping, self.tag_dict = Utils.build_tag_vectors("KGRec-dataset/KGRec-dataset/KGRec-music/tags")
        id_vec_mapping, self.tfidf_model = Utils.build_tfidf_vectors(id_vec_mapping)
        self.lsi_model = LsiModel(id_vec_mapping.values(), id2word=self.tag_dict, num_topics=self.vector_size)
        for song_id, tfidf_vector in id_vec_mapping.items():
            sparse_vec = self.lsi_model[tfidf_vector]
            if len(sparse_vec) > 0:
                self.item_vectors[song_id] = Utils.sparse_to_dense(sparse_vec, self.vector_size, norm=True)

    def build_users(self, train):
        """ builds a user vector by summing the item vectors in user's training set and normalizing to unit length
        Parameters
        ----------
            train : dict, (user_id => training items)
        """
        for user, history in train.items():
            user_vec = np.zeros(self.vector_size)
            for song_id in history:
                user_vec += self.item_vectors[song_id]
            norm_value = np.linalg.norm(user_vec)
            self.user_vectors[user] = user_vec / norm_value


lsiModel = LatentSemanticIndexingModel(vector_size=50)
print("computing item vectors")
lsiModel.compute_vectors()
print("splitting dataset")
train, valid, test = Utils.load_dataset(lsiModel.item_vectors.keys())
print(str(len(lsiModel.item_vectors)) + " items")
print(str(len(test)) + " users")
print("getting recommendations")
recs = lsiModel.get_recs(train, k=10)
print("computing metrics")
hr, ndcg = Utils.compute_metrics(recs, valid)
hr_cs, ndcg_cs = Utils.compute_cold_start_metrics(recs, train, valid)
print(hr)
print(ndcg)
print(hr_cs)
print(ndcg_cs)
