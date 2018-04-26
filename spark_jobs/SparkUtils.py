
def load_dataset(sc, validation=True, valid_songs=None):
    """ Loads training set
        Parameters
        ----------
            sc: Spark Context
            validation : if True, removes last 2 items of each sequence, (one for test set, one for validation)
                if False, removes only last item of sequence, (includes validation set in training)
            valid_songs : set of song_id valid for dataset (invalid songs could be songs with no tags, for example)
                if None, assumes all songs are valid
        Returns
        -------
            data : RDD[sequence of items]
    """
    data = sc.textFile("text_rec_research/implicit_lf_dataset.csv")\
        .map(lambda x: (x.split('\t')[0], x.split('\t')[1]))\
        .zipWithIndex()\
        .map(lambda t: (t[0][0], (t[0][1], t[1])))\
        .groupByKey()
    if validation:
        data = data.map(lambda user: [i[0] for i in sorted(list(user[1]), key=lambda it: it[1])][:-2])
    else:
        data = data.map(lambda user: [i[0] for i in sorted(list(user[1]), key=lambda it: it[1])][:-1])
    if valid_songs:
        data = data.map(lambda sequence: [i for i in sequence if i in valid_songs])
    return data.filter(lambda x: len(x) > 1)


def parse_words(line):
    """ parses the tag list string
        Parameters
        ----------
            line : str, line of item_tags list
        Returns
        -------
            list of tags
    """
    tags = line[line.index('[')+1:line.index(']')].split(',')
    num_tags = len(tags)
    for i in range(num_tags):
        word = tags[i].strip()
        tags[i] = word[1:-1]
    return tags


def load_words(sc):
    """ Loads the item text content
        Parameters
        ----------
            sc : sparkcontext
        Returns
        -------
            dict of (item_id => list of words)
    """
    return sc.textFile("text_rec_research/item_tags.txt")\
        .map(lambda x: (x.split(' ')[0], parse_words(x)))\
        .collectAsMap()


def save_embeddings(sc, vectors, output_dir):
    """ saves learned vectors
        Parameters
        ----------
            sc: sparkcontext
            vectors : dict (itemid => vector)
            output_dir : name of output directory to save vectors
    """
    items = []
    for item, vector in vectors.items():
        prod_vec = [item]
        prod_vec.extend(vector)
        items.append(prod_vec)
    vectors = sc.parallelize(items) \
        .map(lambda x: (x[0], x[1:]))
    vectors.coalesce(1).saveAsTextFile(output_dir)
