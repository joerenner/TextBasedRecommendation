from ML20m.data_processing import load_word_embeddings
from sklearn.neighbors import BallTree


movies = False

if movies:
    word_vectors = load_word_embeddings("embeddings/PW2V_200_2_20_01_test.txt", 200)
    tags = ["war crimes", "western", "love triangle", "disney animated feature", "sad"]
else:
    word_vectors = {}
    with open("../embeddings/word_embeddings100.txt", 'r') as f:
        for line in f:
            item_id = line[line.index("(") + 2:line.index(",") - 1]
            vector = line[line.index("[") + 1:-3].split(",")
            vector = [float(x.strip()) for x in vector]
            word_vectors[item_id] = vector
    tags = ["Sentimental", "folk-rock", "Disney", "1980s", "Swing-Jazz"]

k = 6
X = list(word_vectors.values())
ids = list(word_vectors.keys())
id_index_list = {}
num_items = len(ids)
for i in range(num_items):
    id_index_list[ids[i]] = i
tree = BallTree(X, leaf_size=40)
tag_recs = {}

for tag in tags:
    _, recs_ind = tree.query([word_vectors[tag]], k=k)
    recs = []
    for i in recs_ind[0]:
        # filter items in user history
            recs.append(ids[i])
    tag_recs[tag] = recs

for tag, recs in tag_recs.items():
    print(recs)


