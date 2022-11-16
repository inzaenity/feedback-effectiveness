    
import numpy as np


np.random.seed(1)


def load_embeddings(word_index, max_features, embed_size):
    embedding_file = open(f"glove.6B.{embed_size}d.txt", "r")
    lines = embedding_file.readlines()

    embeddings = dict()
    for line in lines:
        wordvec = line.strip().split(" ")
        word, vec = wordvec[0], wordvec[1:]
        embeddings[word] = [float(x) for x in vec]

    embedding_values = np.stack(list(embeddings.values()))
    mean, std = embedding_values.mean(), embedding_values.std()
    
    matrix = np.random.normal(mean, std, (len(word_index) + 1, embed_size))

    for w, i in word_index.items():
        if i > max_features:
            continue
        if embeddings.get(w) != None:
            matrix[i] = embeddings[w]
        else:
            capitalized_vec = embeddings.get(w.capitalize())
            if capitalized_vec != None:
                matrix[i] = capitalized_vec
    
    return matrix
