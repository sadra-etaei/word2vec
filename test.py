from src.skip_gram_negative_sampling import Word2Vec



model,processor = Word2Vec.load_model("D:/projects/nlp/word2vec_project/brown_word2vec_full.npz",200)

print("Vocab size:", processor.vocab_size)

print(model.solve_analogy('king','man','woman',processor))

print(model.get_similar_words('king',processor))

print(model.get_similar_words('man',processor))