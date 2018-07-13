import re
import gensim
from nltk.corpus import stopwords

class WhiskyClass:
    """
    This class harmonizes the whisky reviews contained in reviews.csv.
    """
    def __init__(self, raw, tokenize=True, rm_stopwords=True):
        self.tokenize = tokenize
        self.rm_stopwords = rm_stopwords
        self.badwords = None
        if self.rm_stopwords:
            self.badwords = stopwords.words('english')

        self.name = raw[0].lower()
        nr_ = len(raw)-1
        self.url = raw[nr_]

        # Separate review dimension and description
        tups = [re.split(': ', x, maxsplit=1) for x in raw[1:nr_]]
        self.att = dict(tups)

        # Select tokenization function (simple split vs gensim preprocessing)
        self.split_n_prep = self.choose_split_n_prep()
        for k, v in self.att.items():
            self.att[k] = self.split_n_prep(v)
            if rm_stopwords:
                self.att[k] = self.clean_badwords(self.att[k])
        self.harmonise_keys()

    def harmonise_keys(self):
        # Synonyms for the whisky tasting dimensions:
        pal_syn = ['tongue', 'tasted', 'pallate', 'palette', 'pallet',
                   'flavour', 'flavor', 'pilate', 'taste', 'tasted',
                   'tasting', 'palatte']
        col_syn = ['color']
        nos_syn = ['sniffling', 'nosing', 'smell', 'sniff', 'supernose',
                   'nosewise', 'aroma', 'noses', 'snout']
        fin_syn = ['dev', 'finally', 'finishing', 'evolution',
                   'swallow', 'afterward', 'finnish']

        # From capitalized to lower case:
        old_keys = list(self.att.keys())
        low_keys = [x.lower().strip() for x in old_keys]
        for old, low in zip(old_keys, low_keys):
            self.att[low] = self.att.pop(old)

            # Pop all synonyms in favour of pal, col, nos or fin:
            if low in pal_syn:
                self.att['pal'] = self.att.pop(low)
                low = 'pal'
            if low in col_syn:
                self.att['col'] = self.att.pop(low)
                low = 'col'
            if low in nos_syn:
                self.att['nos'] = self.att.pop(low)
                low = 'nos'
            if low in fin_syn:
                self.att['fin'] = self.att.pop(low)
                low = 'fin'

            s_low = low[:3]
            self.att[s_low] = self.att.pop(low)
        pass

    def choose_split_n_prep(self):
        if self.tokenize:
            return gensim.parsing.preprocess_string
        else:
            def simple_prep(sometext):
                punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                text = re.compile('[%s]' % re.escape(punct)).sub('', sometext)
                return str(text).lower().strip().split()
            return simple_prep

    def add_badwords(self, words):
        if isinstance(words, str):
            words = [words]
        if self.tokenize:
            joined = ' '.join(stopwords.words('english'))
            words = gensim.parsing.preprocess_string(joined)
        self.badwords += words

    def clean_badwords(self, review):
        return [w for w in review if w not in self.badwords]
