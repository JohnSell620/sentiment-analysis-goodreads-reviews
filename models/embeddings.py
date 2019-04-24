def get_fastText_embedding():
        from gensim.models import FastText
        return FastText.load_fasttext_format('../data/cc.en.300.bin')

def get_elmo_embeddings():
    pass
