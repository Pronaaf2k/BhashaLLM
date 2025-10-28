"""
Word2Vec Spelling Correction Training
Config: Skip-gram, window=4, vector=300, min_count=1, workers=8, epochs=100, sample=0.01
"""
from gensim.models import Word2Vec
import pickle

def train_corrector(text_corpus_path='data/bangla_text_corpus.txt'):
    """
    text_corpus_path: File with Bengali text (one sentence per line)
    Corpus: District names, names, dictionary, news articles
    """
    # Load corpus
    with open(text_corpus_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    
    print(f"Training Word2Vec on {len(sentences)} sentences...")
    
    # Train Word2Vec
    model = Word2Vec(
        sentences=sentences,
        sg=1,              # Skip-gram
        window=4,
        vector_size=300,
        min_count=1,
        workers=8,
        epochs=100,
        sample=0.01
    )
    
    # Save model
    model.save('models/word2vec_corrector.pkl')
    print("Word2Vec model saved to models/word2vec_corrector.pkl")
    
    # Test correction
    test_word = 'খরদ'
    if test_word in model.wv:
        similar = model.wv.most_similar(test_word, topn=3)
        print(f"Similar words to '{test_word}': {similar}")

if __name__ == '__main__':
    # Create sample corpus if needed
    sample_corpus = """
করদ খরদ গড়া গরা
বাংলা ভাষা অক্ষর
হাতে লেখা চিঠি
""".strip()
    
    with open('data/bangla_text_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(sample_corpus)
    
    train_corrector()