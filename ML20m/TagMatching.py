from gensim.corpora import Dictionary
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import data_processing


class TagMatchingModel:

    def __init__(self, movie_tags):
        """
            movie_tags: dict {item_id => tags}
        """
        self.item_vectors = {}
        self.id_to_idx = {}
        self.idx_to_id = []
        dictionary = Dictionary()
        for item, tags in movie_tags.items():
            dictionary.add_documents([tags])
        for item, tags in movie_tags.items():
            self.item_vectors[item] = dictionary.doc2bow(tags)
        data = []
        row_ind = []
        col_ind = []
        i = 0
        for item, tags in self.item_vectors.items():
            for (col, count) in tags:
                data.append(count)
                row_ind.append(i)
                col_ind.append(col)
            self.id_to_idx[item] = i
            self.idx_to_id.append(item)
            i += 1
        self.item_vectors = csr_matrix((data, (row_ind, col_ind)), shape=(i, dictionary.num_pos))

    def get_recs(self, user_history, k):
        tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.item_vectors)
        user_recs = {}
        j = 0
        num_users = len(user_history)
        for user, song_ids in user_history.items():
            _, recs_ind = tree.kneighbors(self.item_vectors.getrow(self.id_to_idx[song_ids[-1:][0]]), n_neighbors=k + len(song_ids))
            recs = []
            for i in recs_ind[0]:
                # filter items in user history
                if self.idx_to_id[i] not in song_ids:
                    recs.append(self.idx_to_id[i])
            user_recs[user] = recs[:k]
            j += 1
            if j % 1000 == 0:
                print(str(j) + " out of " + str(num_users) + " recs computed")
        return user_recs


if __name__ == "__main__":
    print("loading data...")
    train, train_sets, test = data_processing.load_data(validation=False)
    movie_tags, vocab = data_processing.load_tags(train_sets, min_count=2)
    train, test = data_processing.filter_movies_with_no_tags(train, test, movie_tags)
    tagModel = TagMatchingModel(movie_tags)
    print("getting recs...")
    recs = tagModel.get_recs(train, 10)
    hr, ndcg = data_processing.compute_metrics(recs, test)
    print(hr)
    print(ndcg)
    cs_hr, cs_ndcg = data_processing.compute_cold_start_metrics(recs, train, test, window_size=2)
    print(cs_hr)
    print(cs_ndcg)
