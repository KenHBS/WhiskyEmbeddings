import difflib
import gensim
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


class WordEmbeddings:
    """ This class trains a word2vec model based on raw as a corpus. """
    def __init__(self, raw, per_feat=True):
        self.per_feat = per_feat
        if self.per_feat:
            self.corpus = [sen for x in raw for sen in x.att.values() if len(sen) > 5]
        else:
            self.corpus = [sum(x.att.values(), []) for x in raw]

        self.model = None

    def train(self, size, window, skipgram=True, workers=4, min_count=None):
        """
        :param size: (int) The dimension of the word embeddings
        :param window: (int) The window size during word2vec training
        :param skipgram: (bool) default is True. Use skipgram (True) or
            CBOW (False)
        :param workers: (int) default is 4. Number of workers to train model
        :param min_count: (int) The minimum nr of occurrences of a word in the
            corpus
        :return:
        """
        print('Creating word embeddings..')
        model = gensim.models.Word2Vec(self.corpus,
                                       size=size, window=window,
                                       min_count=min_count,
                                       workers=workers,
                                       sg=skipgram)
        self.word_vectors = model.wv
        return model.wv


class WhiskyEmbeddings:
    def __init__(self, raw, w2v):
        self.wv = w2v.wv
        self.wordlist = list(self.wv.vocab.keys())
        self.wv_vecs = np.vstack(self.wv[x] for x in self.wv.vocab.keys())
        self.raw = [i for i in raw if self.is_complete(i)]

        self.names = [x.name for x in self.raw]
        self.count = self.count_dict(self.names)

        _embeddings = self.embed_whiskies()
        self.embeddings = self.aggregate_embeddings(_embeddings)

    @staticmethod
    def is_complete(wr):
        check = ['nos', 'pal', 'fin']
        overlap = [x in check for x in wr.att.keys() if len(wr.att[x]) > 5]
        return sum(overlap) == 3

    @staticmethod
    def count_dict(some_list):
        counts = {}
        for i in some_list:
            try:
                counts[i] += 1
            except KeyError:
                counts[i] = 1
        return counts

    def att2vec(self, att):
        """
        Turns a list of words (an attribute of a whisky review) into their
        individual embeddings and calculates the average of the embeddings,
        thus creating a sentence embedding

        :param att: (list) list of single word strings (a splitted sentence)
        :return: (np.array) the average embeddings of the sentence
        """
        vec_container = []
        for w in att:
            try:
                vec = self.wv[w]
            except KeyError:
                continue
            vec_container.append(vec)
        return np.mean(vec_container, axis=0)

    def embed_whisky(self, wr):
        """
        Takes a whisky review (wr) and calculates the 'sentence embedding' of
        its nose, palate, finish review. The full whisky embedding is the
        average of nose, palate, finish embedding.

        The whisky name and attribute are also returned for later registration
        :param wr: (WhiskyReview)
        :return: list of embeddings
        """
        nos = self.att2vec(wr.att['nos'])
        pal = self.att2vec(wr.att['pal'])
        fin = self.att2vec(wr.att['fin'])
        ful = np.mean([nos, pal, fin], axis=0)

        name = wr.name
        nos = [name, 'nos'] + list(nos)
        pal = [name, 'pal'] + list(pal)
        fin = [name, 'fin'] + list(fin)
        ful = [name, 'ful'] + list(ful)

        return [nos, pal, fin, ful]

    def embed_whiskies(self):
        print('Calculating all whisky embeddings...')
        cont = []
        for wr in self.raw:
            embedded = self.embed_whisky(wr)
            cont.extend([x for x in embedded])

        size = self.wv.vector_size
        cols = ['name', 'att'] + list(range(size))
        return pd.DataFrame(cont, columns=cols)

    @staticmethod
    def aggregate_embeddings(stacked):
        return stacked.groupby(['name', 'att']).mean()

    def most_similar_whiskies(self, whisky, focus='ful', n=15, min_count=3):
        whisky = whisky.lower()

        # Only focus on rows with appropriate focus (nos, pal, fin, ful)
        df = self.embeddings.xs(focus, level='att')

        try:
            target = df.ix[whisky]
        except KeyError:
            whisky = self.correct_typo(whisky)
            if whisky is None:
                return None
            target = df.ix[whisky]
        target = target.values.reshape(1, -1)

        # Only return whiskies that have an appropriate nr of reviews:
        keep = [self.count[x] >= min_count for x in df.index]
        result_set = df.loc[keep, :]

        # Calculate the distance in embedding of whisky w.r.t. all whiskies
        # and return the n closest ones:
        distances = cosine_similarity(target, result_set)
        inds = np.argsort(distances)[0]
        inds = inds[::-1][1:n]
        for x in inds:
            print([result_set.index[x], distances[0, x]])
        pass

    def correct_typo(self, whisky):
        """
        Find the next best whisky name. Take a typo whisky
        :param whisky: (int) a non-present whisky in database
        :return: (int) the most likely whisky that is present in database
        """
        print('%s is not in the database' % whisky)
        w_names = self.names
        matches = list(filter(lambda x: bool(re.search(whisky, x)), w_names))
        if len(matches) == 0:
            matches = difflib.get_close_matches(whisky, w_names)

        cnt = [self.count[x] for x in matches]
        try:
            m = max(cnt)
        except ValueError:
            print('could not find a suitable match, try some other name')
            return None
        ind = [i for i, j in enumerate(cnt) if j == m][0]

        new_name = matches[ind]
        print('these are the results for %s' % new_name)
        return new_name

    def describe_whisky(self, whisky, n=10):
        df = self.embeddings
        try:
            target = df.xs(whisky, level='name')
        except KeyError:
            whisky = self.correct_typo(whisky)
            target = df.xs(whisky, level='name')

        nos = target.ix['nos']
        pal = target.ix['pal']
        fin = target.ix['fin']

        topnos = self.compare_to_vocab(nos, n=n)
        toppal = self.compare_to_vocab(pal, n=n)
        topfin = self.compare_to_vocab(fin, n=n)

        print('nose:')
        print(topnos)
        print('palate:')
        print(toppal)
        print('finish:')
        print(topfin)

        pass

    def compare_to_vocab(self, embedding, n=10):
        embedding = embedding.values.reshape(1, -1)
        distances = cosine_similarity(embedding, self.wv_vecs)[0]

        inds = np.argsort(distances)
        inds = inds[::-1][1:n]

        vocab = self.wordlist
        return [vocab[x] for x in inds]
