from embeddings import WordEmbeddings, WhiskyEmbeddings
from dataclass import WhiskyClass as WC
import csv


data = []
with open('reviews.csv', 'rt') as rv:
    reader = csv.reader(rv, delimiter=',')
    for line in reader:
        data.append(line)


def run_it_all(dat, tok, rm_s, size, window, skipgram, workers, min_count):
    """
    Return a WhiskyEmbeddings object which allows for some cool trickeries
    such as finding similar whiskies, describing whiskies and finding
    similar wordings ('synonyms') in the whisky-tasting vocabulary.

    This function does it all from beginning to the end:
    1) Transform the scraped whisky reviews into a well-structured object
    2) Use all whisky reviews to build a corpus and train a whisky-specific
        word2vec model.
    3) Use the word embeddings to create whisky embeddings.

    The methods in WhiskyEmbeddings can then be used.
    All of this takes approx. 30-60 seconds.

    :param dat: input data
    :param tok: (bool) use tokenize and gensim preprocessing or not?
    :param rm_s: (bool) remove stopwords or not?
    :param size: (int) the number of word2vec dimensions
    :param window: (int) window size of the context while training word2vec
    :param skipgram: (bool) use skipgram model or CBOW?
    :param workers: (int) number of workers to train word2vec
    :param min_count: (int) min. number of occurrences in corpus. Words with
        less occurences will be deleted.
    :return: (WhiskyEmbeddings instance)
    """
    # 1) Transform whisky reviews into well-structured objects:
    all_reviews = [WC(x, tokenize=tok, rm_stopwords=rm_s) for x in dat[1:]]

    # 2) Build a corpus and train a word2vec model:
    w2v = WordEmbeddings(all_reviews)
    word_vectors = w2v.train(size, window, skipgram, workers, min_count)

    # 3) Create whisky embeddings
    w_embedding = WhiskyEmbeddings(all_reviews, word_vectors)
    return w_embedding


we1 = run_it_all(data, tok=True, rm_s=True, size=50, window=3,
                 skipgram=True, workers=5, min_count=30)
we2 = run_it_all(data, tok=True, rm_s=True, size=100, window=3,
                 skipgram=True, workers=5, min_count=30)
we3 = run_it_all(data, tok=True, rm_s=True, size=100, window=3,
                 skipgram=True, workers=5, min_count=50)


we1.most_similar_whiskies('coal ila 12')

we1.describe_whisky('lagavulin 16', n=20)
we2.describe_whisky('lagavulin 16', n=20)

we1.describe_whisky('ardbeg 10', n=20)
we2.describe_whisky('argbed 10', n=20)


## Tokenization improves results!
##
