# Whisky Embeddings

This project is closely related to [this whisky project](http://wrec.herokuapp.com/methodology).

About 25.000 whisky reviews were scraped from reddit (see data_collection).

After training a word2vec model on this whisky corpus, whisky embeddings are created and 
some neat tricks can be applied to these vectors that describe a single whisky. 

We can:
    - find the most similar whiskies to, say, Highland Park 12.
        A simple check shows that the model does a pretty good job. 
        If you take, for example, an american rye whisky, the ten closest whiskies are 
        also rye whiskies. Similar results show for e.g. heavily peated Islay whiskies. 
    - find synonymous words in the whisky tasting vocabulary. ('tangerin' = 'clementin')
    - describe your favourite whisky in terms of smell ('nose'), taste ('palate') and
        finish. 
        
A more elaborate outline of this project will soon follow soon, including how to reproduce
the results and a more technical perspective.
