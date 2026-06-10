import nltk
from src.processor import TextProcessor
from src.skip_gram_negative_sampling import Word2Vec



brown = nltk.download('brown')
from nltk.corpus import brown



corpus = brown.words()


processor = TextProcessor()



token_ids=processor.build_vocab(corpus)

model = Word2Vec(processor.vocab_size,200,0.06)

model.fit(processor,token_ids,128,5)

model.save_model("brown_word2vec_full.npz", processor)