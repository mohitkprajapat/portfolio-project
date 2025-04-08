import numpy as np
from tensorflow.keras.layers import Embedding

def read_glove_vecs(glove_file):    
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map



def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):
        sentence_words = X[i].lower().split(' ')
        j = 0
        for w in sentence_words:
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
                j +=  1    
    return X_indices


def indices_to_sentences(X, index_to_words):
    sentences = []
    
    for i in X:
        sentence_i = ""
        for j in i:
            if j == 0:
                break
            else:
                sentence_i += index_to_words[j]
                sentence_i += " "
        sentences.append(sentence_i[:-1])
    
    return sentences



def pretrained_embedding_layer(word_to_index,word_to_vec_map):
    
    vocab_size = len(word_to_index) + 1
    any_word = next(iter(word_to_vec_map.keys()))
    emb_dim = word_to_vec_map[any_word].shape[0]

    emb_matrix = np.zeros((vocab_size,emb_dim))
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_size,emb_dim,trainable = False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

