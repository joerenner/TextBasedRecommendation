from KNearestNeighborsRec import KNearestNeighborsRecModel
import data_processing


class ContentEmbToRec(KNearestNeighborsRecModel):

    def __init__(self, content_embed_file, embedding_size, movie_tags, weights, words=True):
        """
            content_embed_file: pretrained word embeddings file name
            embedding_size: vector size
            movie_tags: dict of {movieId => list of tags}
            weights:
                None : uniform weighting for each word
                "tfidf" : compute tfidf for each product tags, weight by value
                vector of shape (vocab_size) : use learned word weights
                words : Boolean, True: using word embeddings, False: product embeddings
        """
        embeddings = data_processing.load_word_embeddings(content_embed_file, embedding_size)
        if words:
            self.item_vectors = data_processing.build_product_embeddings(embeddings, movie_tags, weights=weights)
        else:
            self.item_vectors = embeddings


words = True
weights = False
print("loading data...")
train, train_sets, test = data_processing.load_data(validation=False)
movie_tags = None
if words:
    movie_tags, vocab = data_processing.load_tags(train_sets, min_count=2)
    train, test = data_processing.filter_movies_with_no_tags(train, test, movie_tags)
if weights:
    weights = data_processing.load_word_weights("embeddings/PW2V_200_2_20_01_test.txt")
print("getting recs...")
word_embed_to_recs = ContentEmbToRec("embeddings/PW2V_200_2_20_01_test.txt", 200, movie_tags, "tfidf", words)
recs = word_embed_to_recs.get_recs(train, 10)
hr, ndcg = data_processing.compute_metrics(recs, test)
print(hr)
print(ndcg)
cs_hr, cs_ndcg = data_processing.compute_cold_start_metrics(recs, train, test, window_size=2)
print(cs_hr)
print(cs_ndcg)
