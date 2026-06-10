import numpy as np
import os
from src.processor import TextProcessor

            


class Word2Vec:
    def __init__(self,vocab_size,embed_dim,lr):
        self.V =vocab_size
        self.D=embed_dim
        self.lr = lr

        self.W_target = np.random.uniform(-0.5/embed_dim,0.5/embed_dim,(self.V,self.D))
        self.W_context = np.zeros((self.V,self.D))

        self.G_target = np.zeros((self.V,self.D))
        self.G_context = np.zeros((self.V,self.D))


        self.eps=1e-6


    def sigmoid(self,x):
        x = np.clip(x,-10,10)
        return 1 / ( 1 + np.exp(-x) )
    

    def train(self,target_ids,context_ids,neg_samples):

        B = len(target_ids)
        K = neg_samples.shape[1]


        v = self.W_target[target_ids]

        all_context = np.zeros((B, K + 1), dtype=np.int64)
        
        
        all_context[:, 0] = context_ids        
        all_context[:, 1:] = neg_samples

        u = self.W_target[all_context]


        scores = np.einsum('bd,bkd->bk',v,u)
        predictions = self.sigmoid(scores)

        labels = np.zeros((B,K+1))
        labels[:,0] = 1.0

        predictions_clipped = np.clip(predictions,1e-10,1.0 - 1e-10)

        batch_loss = -np.sum(labels*np.log(predictions_clipped) + (1.0-labels)*np.log(1-predictions_clipped))/B


        errors = predictions - labels 


        grad_W_target_batch = np.einsum('bk,bkd->bd',errors,u)

        grad_W_context_batch = np.einsum('bk,bd->bkd',errors,v)


        grad_W_target = np.zeros_like(self.W_target)

        np.add.at(grad_W_target,target_ids,grad_W_target_batch)


        self.G_target += grad_W_target**2

        self.W_target -= (self.lr/ (np.sqrt(self.G_target)+self.eps) ) * grad_W_target


        grad_W_context = np.zeros_like(self.W_context)

        np.add.at(grad_W_context,all_context,grad_W_context_batch)

        self.G_context += grad_W_context**2

        self.W_target -= (self.lr/ (np.sqrt(self.G_context)+self.eps) ) * grad_W_context


        return batch_loss
    
    def fit(self,processor,token_ids,batch_size,epochs):



        print(f"Initializing Fit Sequence | Vocab Size: {self.V} | Embed Dim: {self.D}")
        print(f"Running across {epochs} Epoch(s) with Batch Size: {batch_size}")
        print("-" * 75)


        for epoch in range(epochs):
            total_loss = 0.0
            step = 0

            batch_generator = processor.generate_batches(token_ids,batch_size)


            for targets,contexts,negs in batch_generator:
                loss = self.train(targets,contexts,negs)

                total_loss+=loss

                step+=1

                if step % 5 == 0:
                    print(f"Epoch {epoch+1:02d} | Step {step:03d} | Batch Loss: {loss:.4f} ")
            
            mean_loss = total_loss / max(1, step)
            print(f"==> Epoch {epoch+1:02d} Complete | Mean Epoch Loss: {mean_loss:.4f}\n")

        
    def get_similar_words(self, word, processor, top_n=3):
        if word not in processor.word2id:
            print(f"Word '{word}' not found in vocabulary.")
            return []
            
        target_id = processor.word2id[word]

        self.W_combined = self.W_target + self.W_context
        
        v_combined = self.W_combined[target_id] 
        
        
        eps = 1e-8
        W_combined_norms = np.linalg.norm(self.W_combined, axis=1, keepdims=True) + eps
        combined_norm = np.linalg.norm(v_combined) + eps
        W_target_normalized = self.W_combined / W_combined_norms
        v_target_normalized = v_combined / combined_norm 
        
        similarity_scores = np.dot(W_target_normalized, v_target_normalized) 
        
      
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        
        results = []
        for idx in sorted_indices:
            if idx == target_id:
                continue
            if len(results) >= top_n:
                break
                
            similar_word = processor.id2word[idx]
            score = similarity_scores[idx]
            results.append((similar_word, score))
            
        return results
    

    def save_model(self, filepath, processor):
        np.savez_compressed(
            filepath,
            W_target=self.W_target,
            W_context=self.W_context,
            G_target=self.G_target,
            G_context=self.G_context,
            vocab_words=np.array(list(processor.word2id.keys())),
            vocab_ids=np.array(list(processor.word2id.values()))
        )   
        print(f"Successfully saved trained model state to: {filepath}")

    @staticmethod
    def load_model(filepath, embed_dim=100, initial_lr=0.01):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
    
        processor = TextProcessor()
        words = data['vocab_words']
        ids = data['vocab_ids']
        processor.word2id = {str(w): int(i) for w, i in zip(words, ids)}
        processor.id2word = {int(i): str(w) for w, i in zip(words, ids)}
        processor.vocab_size = len(processor.word2id)
    
        model = Word2Vec(
            vocab_size=processor.vocab_size,
            embed_dim=embed_dim,
            lr=initial_lr
    )
        model.W_target = data['W_target']
        model.W_context = data['W_context']
        model.G_target = data['G_target']
        model.G_context = data['G_context']
    
        print(f"Successfully loaded trained model and vocabulary from {filepath}!")
        return model, processor
    
    def solve_analogy(self, word_a, word_b, word_c, processor, top_n=3):
        # 1. Clean inputs and ensure all words exist in the vocabulary
        words = [word_a, word_b, word_c]
        for w in words:
            if w not in processor.word2id:
                print(f"Error: Word '{w}' not found in vocabulary.")
                return []
                
        id_a = processor.word2id[words[0]]  
        id_b = processor.word2id[words[1]]  
        id_c = processor.word2id[words[2]]  
    

        self.W_combined = self.W_target + self.W_context

        v_a = self.W_combined[id_a]
        v_b = self.W_combined[id_b]
        v_c = self.W_combined[id_c]
        
        v_combined = v_a - v_b + v_c  # Shape: (D,)
        
        eps = 1e-8
        W_target_norms = np.linalg.norm(self.W_target, axis=1, keepdims=True) + eps
        target_norm = np.linalg.norm(v_combined) + eps
        
        W_combined_normalized = self.W_target / W_target_norms
        v_combined_normalized = v_combined / target_norm
        
        # 4. Compute similarity scores across the whole vocabulary instantly
        similarity_scores = np.dot(W_combined_normalized, v_combined_normalized) # Shape: (V,)
        
        # 5. Sort indices by highest score
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        # 6. Gather top results, filtering out the input words to prevent cheap matches
        input_ids = {id_a, id_b, id_c}
        results = []
        
        for idx in sorted_indices:
            if idx in input_ids:
                continue
            if len(results) >= top_n:
                break
                
            word = processor.id2word[idx]
            score = similarity_scores[idx]
            results.append((word, score))
            
        return results
    

























