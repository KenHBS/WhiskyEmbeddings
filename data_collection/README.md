# Scrape Reddit Whisky Reviews
The corpus of 25k whisky reviews is built on [this document](https://docs.google.com/spreadsheets/d/1X1HTxkI6SqsdpNSkSSivMzpxNT-oeTbjFFDdEkXD30o/edit#gid=695409533). I have retained the relevant information, and save it in `named_urls.csv`.  

To obtain all whisky reviews, you need to prepare two things:
1. get credentials to the Reddit API and use PRAW (Python Reddit API Wrapper)  and save them at `path1`. See `dummycredits.ini` for an example of how what you need. Check [here](https://praw.readthedocs.io/en/latest/getting_started/quick_start.html) and [here](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps) for instruction on how to get the proper authentication.
2. download `named_urls.csv` and save it at `path2`

Then, it is as simple as opening the terminal and run
```
$ python3 scrape_reviews.py -doc1 path1 -doc2 path2
```
A new file called `reviews.csv` will be added to your home directory. 

It will also produce a file called `phony_urls.csv`, which will contain the URLs that somehow ended up throwing errors. I added this investigate common sources of errors, but have not dug into it yet. 
