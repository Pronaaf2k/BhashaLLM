import os
import json
import pickle
import re 

text_file_name = "data/bangla_words.txt"
saved_file_name = "data/bn_words_corpus.pickle"

'''
This script is going to make a list of sentences from a raw text file.
Finally, it will save the file as pickle format in the data folder as "bn_corpus.pickle"
'''
def preprocess():

    with open(text_file_name, "r", encoding="utf-8") as f:
        text = f.read()

    # Split sentences by both '।' and '?'
    sentences = re.split('।|\?', text)

    # Remove '\n' from the sentences
    sentences = [sen.replace("\n", " ") for sen in sentences]

    print(f"\nTotal Bangla sentences: {len(sentences)} ")

    with open(saved_file_name, "wb") as files:
        pickle.dump(sentences, file=files)
    print(f"bn_corpus file saved in the data/ folder ")

if __name__ == '__main__':
    preprocess()
