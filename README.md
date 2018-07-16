
# Whisky Embeddings - Introduction

This project is a reproduction of [this whisky project]([http://wrec.herokuapp.com/methodology](http://wrec.herokuapp.com/methodology)).

The main idea of the project is to literally put numbers on the characteristics of different whiskies. If we can represent every whisky as a single dot in a, say, 100 dimensional space, we should also be able to find whiskies that are located close to it, right?
```python

>>> w_embeddings.most_similar_whiskies('caol ila 12')
['kilchoman machir bay', 0.9965]
['ardbeg 10', 0.9961]
['lagavulin 12', 0.9959]
['laphroaig 18', 0.9953]
['talisker 10', 0.9951]
['lagavulin 16', 0.9949]
['talisker storm', 0.9945]
...
```
For the kind-of-informed readers, this is pretty accurate. Based on a similar notion of distance/similarity, some additional features of whiskies will be exposed along the way, too.

The project roughly consists of four components:

1. Data collection via Reddit API
2. Structure the whisky reviews & corpus preparation
3. Train [word2vec]([https://en.wikipedia.org/wiki/Word2vec](https://en.wikipedia.org/wiki/Word2vec)) model with [gensim]([https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html))
4. Create whisky embeddings and related methods

## 1. Data collection

`data_collection` contains a dataset with urls to 25k reddit reviews. See its [readme](data_collection/README.md) for more information. For an easy start, I have included `dummydata.csv`, too. This contains only about 10% of the actual data, but allows you to run the remainder of this repository.

## 2. Structure review & corpus preparation

A whisky review should discuss at least three attributes of a whisky: its smell (nose), its taste (palate) and its finish. Usually, there will be some general free text, too. Your average review may look something like this:

> (..) Ardbeg 10 (..)
Nose: Light, fresh fruit - sour red apple, pear, bitter lemon peel. Some vanilla butter (think Corryvreckan Light), an unmistakable maritime note, dry and fresh mentholy peat smoke. Lemon shortcake. Water brings out more citrus and a bit of brine.
Taste: Very balanced despite the smoke promised in the nose. Thick, oily mouthfeel for a young 46%er. Smoky malty sweetness, pepper, pineapple, ashy. Water brings out estery pear.
Finish: This is where mister smoke makes his big entrance. Peat explosion. Some pepper, butter, tiny bit of oak. Long finish, some prickling spices towards the end.
Score: For being so widely available and their entry level dram, this is fantastic. I always thought I preferred the Frog but just the fact that Ardbeg seems to care for the craft by making it 46% NCNCF makes me like this more. 85/100.

`dataclass.py` introduces `WhiskyClass` that structures each review. It also provides methods to preprocess the corpus such as stopword removal and different types of tokenisation. Also, the reviews are semi-structured, so we have to make sure that 'smell', 'nose', 'sniffing', 'aroma' are picked up to refer to the same thing.

```python
from dataclass import WhiskyClass as WC
import csv

data = []
with open('reviews.csv', 'rt') as rv:
reader = csv.reader(rv, delimiter=',')
for line in reader:
data.append(line)

all_reviews = [WC(x, tokenize=True, rm_stopwords=True) for x in data[1:]]

one_review = all_reviews[1204]
one_review.name     # Ardbeg 10
one_review.att		# {'col': ['pale', 'straw'], 'nos': ['medicin', 'smell', 'bail', 'hai', 'cedar', 'wood', 'vanilla', 'extract', 'salt', 'caramel', 'anis'], 'pal': ['peati', 'salt', 'caramel', 'slight', 'pepper', 'grassi'], 'fin': ['lemon', 'zest', 'oliv', 'brine', 'pepper', 'candi', 'ginger'], 'ove': ['bottl', 'islai', 'shock', 'like', 'bottl', 'nearli', 'month', 'peat', 'medicin', 'note', 'face', 'sweet', 'underton', 'stuff', 'amaz', 'get', 'review']}
```
## 3. Train word2vec model
Training the word2vec model and chosing its parameters is done in `embddings.py`.  This contains two classes: `WordEmbeddings` and `WhiskyEmbeddings` (we'll discuss the latter in point 4).

**The aim of word2vec** is to find a numeric representation of words based on its meaning, or actually, based on the context in which each word is used. If you are unfamiliar with word2vec, check out [the original paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), or the more accessible [NIPS presentation](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit). If you are looking for an extremely quick illustration of the idea behind word2vec's output, check slides 23-25 of the presentation.

`WordEmbeddings` takes the corpus as input, as well as `per_feat`. This prepares the corpus for gensim word2vec training. 

`per_feat` deals with the decision on what a 'sentence' is. Should we regard each attribute in the review as a separate sentence, or is it more appropriate to take an entire review as one sentence? Since word2vec models words in relation to their context, it is important to figure out where a context starts and ends. Setting `per_feat=False` combines an entire review in a single 'sentence' (context).   

After creating an instance of `WordEmbeddings`, the word2vec model can be trained using its `train()` method, which just calls the `gensim.model.Word2Vec`. I've added a brief docstring, but more information can be found [here](https://radimrehurek.com/gensim/models/word2vec.html). 

This is where the first whisky-related feature comes in. Whisky reviewers like to use exotic words for certain flavours that may actually not be too distinctive from each other. We can find closely related words, like this:

```python
from embeddings import WordEmbeddings
w2v = WordEmbeddings(all_reviews)
word_vectors = w2v.train(size=100, window=5, skipgram=True, workers=4, min_count=30)
word_vectors.most_similar('cashew') # [('pistachio', 0.822225034236908), ('arugula', 0.8156542181968689), ('pralin', 0.8154702186584473), ('macadamia', 0.8076515197753906), ('chipotl', 0.7951531410217285), ('walnut', 0.783086359500885), ...]
```

After retaining the word embeddings for each word in the vocabulary, we can move on to the embedding whiskies!

## 4. Whisky Embeddings
Now that each word is represented as an N-dimensional datapoint (N=100 in the example above), we can actually get the meaning of a sentence, too! Taking the mean (or sum) of all word embeddings in a sentence, will give an N-dimensional datapoint, which represents the meaning of the entire sentence. This idea is illustrated on slides 19-21 of the aforementioned [NIPS presentation](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit). 

As such, we can take each attribute of each review and translate it into an N-dimensional datapoint. While doing so, I have only retained the reviews that I consider complete, being the ones that contain more than 5 words on each one of the important attributes (nose, palate, finish). Each review thus got 3 N-dimensional vectors and a summary vector, which is simply the mean of nose, palate and finish. 

Upon translating the reviews containing strings to numeric vectors, the whiskies of the same kind are aggregated together, taking their mean. 

Now we have all the pieces in place to **start finding similar whiskies**. In the introduction you saw the results for a typical peated Islay whisky, and now we'll check out what the closest whiskies are for, say, a typical bourbon called Maker's Mark :
```python
from embeddings import WhiskyEmbeddings
w_embedding = WhiskyEmbeddings(all_reviews, word_vectors)
w_embedding.most_similar_whisky('makers mark')
['w.l. weller special reserve', 0.995735103104071]
['w.l. weller 12', 0.9937612366890741]
['elijah craig 12', 0.9919588142071638]
['colonel e.h. taylor single barrel', 0.9919104918919236]
['four roses yellow label', 0.9909651898226056]
['bulleit bourbon', 0.9893490331607152]
['old ezra 101', 0.9893282718905534]
['old grand dad 86', 0.9891075512787645]
['nikka coffey grain', 0.9890869939430709]
['old weller antique 107', 0.9890158923058934]
['elijah craig 12 ', 0.9889696871529081]
['buffalo trace', 0.9889620787485249]
['knob creek 9 small batch', 0.9889094148804475]
['eagle rare 10', 0.9887739454807991]
```
All of them, except *nikka coffey grain* are bourbons. Nikka is actually Japanese, and this the coffey grain is made from corn, just like bourbon. Just like Maker's Mark, a main feature of Nikka Coffey Grain is its vanilla sweetness. Seems like the whisky embeddings are doing a pretty solid job here. 

By default, `most_similar_whisky()` compares whiskies on based on the average of nose, palate and finish. We could also compare them based on the nose (nos), palate (pal) or finish (fin) like `w_embeddings.most_similar_whisky('aberlour 18', focus='fin')`.

**Describing a whisky** also works out pretty well, especially if there is general consensus on the most important features of a whisky. Since the nose, palate and finish of each whisky are now expressed in the same way as the word embeddings, we can actually check out which words are the closest to whisky:

```python
w_embedding.describe_whisky('makers mark')
nose:
['peek', 'vanillin', 'yam', 'clementin', 'confectionari', 'cornbread', 'bourboni', 'overton', 'scone']
palate:
['raisini', 'bourboni', 'saccharin', 'rosewat', 'gingeri', 'overton', 'cornbread', 'vanillin', 'confectionari']
finish:
['peek', 'unev', 'saccharin', 'fleet', 'lengthi', 'taper', 'raisini', 'chocolati', 'durat']

w_embedding.describe_whisky('aberlour 18')
nose:
['creamsicl', 'rosewat', 'clementin', 'scone', 'muscovado', 'pod', 'lilac', 'croissant', 'milkshak']
palate:
['amaretto', 'pod', 'peek', 'creamsicl', 'vanillin', 'muscovado', 'peatsmok', 'rosewat', 'croissant']
finish:
['raisini', 'mouthwat', 'peek', 'chocolati', 'unev', 'lengthi', 'tingli', 'tanin', 'durat']
```
Surely, it is to be expected that the descriptions are very general, because the whisky reviews have been averaged among many different reviews. Therefore, we may expect that similar whiskies can actually not be distinguished based on the output of `describe_whisky()`. However, it is still pretty cool, IMHO.

## Bonus Material
Especially with the scottish whiskies, it happens that you may mix up some letters like with 'Coal Ila 12' and 'Caol Ila 12'. Or that you simply do not know the exact name or spelling of a whisky -  Bunnahabhain Eirigh na Greine anyone? A simple solution was to create **`correct_typo()`**, which searches for similar names and/or whiskies containing that name as a substring. This made searching and comparing whiskies so much easier and reduces the dreaded `KeyError` to a minimum. 

## To Do's
1. Some visualisations of the whisky embeddings in 2D. 
2. More refined stopword removal, so that only informative and whisky-related terms are considered for word2vec
3. Refine the dataset and corpus building
4. Try to analyse whisky review videos from e.g. Ralfy on YouTube.

