import os
from gensim.models import Word2Vec
from train import EpochSaver
import pickle

def export_weight():
    '''
    Model weight will be saved in the results folder
    '''
    model_path = "results/word2vec_new.model"
    model = Word2Vec.load(model_path)
    with open("results/pretrained_weights.pickle", "wb") as file:
        pickle.dump(model.wv.vectors, file)

    print(f"Pretrained weights successfully saved.")

# exporting weights
if __name__ == '__main__':
    export_weight()
