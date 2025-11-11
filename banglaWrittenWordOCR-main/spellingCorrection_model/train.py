import os
import json
import pickle
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re

remove_chars_pattern = r"[',\":;()@‘“–.\-_]"  # Add this line to define the pattern

'''
Configuration for training
'''
sg = 1
window = 4
size = 300  # Replace 'size' with 'vector_size'
min_count = 1
workers = 8
iters = 100
sample = 0.01

checkpoint = False

os.makedirs("results", exist_ok=True)


# Callback function for saving the model after every epoch
class EpochSaver(CallbackAny2Vec):
    '''Callback to save the model after each epoch.'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

        os.makedirs(self.path_prefix, exist_ok=True)

    def on_epoch_end(self, model):
        saved = "./checkpoints/epoch{}".format(self.epoch)
        model.save(saved)
        print(
            "Epoch saved: {}".format(self.epoch + 1),
            "Start next epoch"
        )
        self.epoch += 1

# Training starts from here
def Train(checkpoint=True):
    '''
    Default checkpoint is true.
    Model will be saved after every epoch
    '''
    with open("data/bn_words_corpus.pickle", "rb") as f:
        data = pickle.load(f)

    # Preprocess the text data to remove specified characters
    data = [re.sub(remove_chars_pattern, '', txt) for txt in data]

    train_data = [txt.split(" ") for txt in data]
    print(len(train_data))

    del data

    if checkpoint:
        model = Word2Vec(train_data, sg=sg, window=window, size=size,
                    min_count=min_count, workers=workers, iter=iters, sample=sample,
                    callbacks=[EpochSaver("./checkpoints")])
        model.save("./results/word2vec_new.model")
        print(f"Training Completed. File saved as 'word2vec_new' in the results folder ")
    else:
        model = Word2Vec(train_data, sg=sg, window=window, size=size,
                    min_count=min_count, workers=workers, iter=iters, sample=sample)
        model.save("./results/word2vec_new.model")
        print(f"Training Completed. File saved as 'word2vec_new' in the results folder ")

if __name__ == '__main__':
    Train(checkpoint)
