from collections import Counter
import numpy as np


class TextProcessor:
    def __init__(self,window_size=2,negative_samples=5):
        self.window_size=window_size
        self.neg_samples = negative_samples
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.unigram_table = []

    # def tokenize(self,text):
    #     return text.lower().split()
    

    def build_vocab(self,tokens,treshold=1e-3):
        word_counts = Counter(tokens)
        count = len(tokens)

        # subsampling frequent words 

        def discard_prob(word_count):
            discard = 1 - np.sqrt((count*treshold)/word_count)
            discard = max(0.0,discard)
            return discard

        keep_prob = {w: 1.0 - discard_prob(c)  for w,c in word_counts.items()}

        filtered_tokens=[w for w in tokens if np.random.random() < keep_prob.get(w,1)]

        unique_words = sorted(list(set(filtered_tokens)))
        self.word2id = {word:i for i,word in enumerate(unique_words)}
        self.id2word = {i:word for i,word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)

        # creating unigram table for negative sampling

        filtered_counts = Counter(filtered_tokens)

        counts_powered = np.array([filtered_counts[self.id2word[i]]**0.75 for i in range(self.vocab_size)])

        self.unigram_table = counts_powered / np.sum(counts_powered)


        return [self.word2id[word] for word in filtered_tokens]
    

    def generate_batches(self, token_ids, batch_size):
        targets = np.zeros(batch_size, dtype=np.int64)
        contexts = np.zeros(batch_size, dtype=np.int64)
        negs = np.zeros((batch_size, self.neg_samples), dtype=np.int64)
        
        idx = 0  # Pointer tracking current row position inside the batch
        
        for i, target_id in enumerate(token_ids):

            current_window = np.random.randint(1,self.window_size+1)
            start = max(0, i - current_window)
            end = min(len(token_ids), i +current_window + 1)
            context_ids = [token_ids[j] for j in range(start, end) if j != i]
            
            for context_id in context_ids:
                neg_samples = np.random.choice(self.vocab_size, size=self.neg_samples, p=self.unigram_table)
                
                
                targets[idx] = int(target_id)
                contexts[idx] = int(context_id)
                negs[idx, :] = neg_samples
                
                idx += 1
                
                # 3. Once our pre-allocated array is full, yield it
                if idx == batch_size:
                    yield targets, contexts, negs
                    
                    # Reset memory layout allocations for the next block
                    targets = np.zeros(batch_size, dtype=np.int64)
                    contexts = np.zeros(batch_size, dtype=np.int64)
                    negs = np.zeros((batch_size, self.neg_samples), dtype=np.int64)
                    idx = 0
                    
        if idx > 0:
            yield targets[:idx], contexts[:idx], negs[:idx, :]