import matplotlib.pyplot as plt
import numpy as np
from src.skip_gram_negative_sampling import Word2Vec



class PCA:
    def __init__(self,n_components=2):
        self.n_components = n_components
        self.components = []
        self.mean =0

    def transform(self,X):
        self.mean = np.mean(X,axis=0)
        X_centered = X - self.mean
        
        
        covariance_matrix = np.dot(X_centered.T,X_centered)/(X.shape[0]-1)

        eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors=eigenvectors[:,sorted_indices]
        self.components = eigenvectors[:,:self.n_components]

        return np.dot(X_centered,self.components)


def plot_embeddings(model, processor, words_to_plot):
    vectors = []
    valid_words = []
    
    # 1. Gather vectors only for words that exist in the trained vocabulary
    for word in words_to_plot:
        cleaned = word.lower().strip()
        if cleaned in processor.word2id:
            word_id = processor.word2id[cleaned]
            vectors.append(model.W_target[word_id])
            valid_words.append(cleaned)
            
    if len(vectors) < 2:
        print("Not enough words found in the vocabulary to visualize.")
        return
        
    X = np.array(vectors) # Shape: (Num_Valid_Words, Embed_Dim)
    
    # 2. Execute our from-scratch PCA dimensionality reduction
    pca = PCA(n_components=2)
    X_2d = pca.transform(X) # Shape: (Num_Valid_Words, 2)
    
    # 3. Render the Matplotlib scatter map
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], color='#4A90E2', edgecolors='k', s=80, alpha=0.8)
    
    # 4. Annotate each plotted coordinates point with its corresponding string label
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(X_2d[i, 0], X_2d[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            fontsize=10,
            weight='bold' if i % 2 == 0 else 'normal',
            alpha=0.9
        )
        
    plt.title("2D PCA Projection of Custom Word2Vec Embeddings", fontsize=14, pad=15, weight='bold')
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Professional touch: center axes lines at 0
    plt.axhline(0, color='grey', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='grey', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.show()



target_vocabulary_sample = [
    "fox", "dog", "brown", "lazy", "cat", "horse",
    "machine", "learning", "computer", "network", "data", "science", "code",
    "language", "processing", "read", "scratch", "wrote", "quick", "jumps"
]

model,processor = Word2Vec.load_model("brown_word2vec_full.npz",200)

plot_embeddings(model,processor,target_vocabulary_sample)