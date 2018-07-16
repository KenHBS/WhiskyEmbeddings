import argparse
import pandas as pd
import praw
import csv
import re


def scrape(path1, path2):
    """
    Uses the Reddit API to get whisky reviews listed in 'named_urls.csv' which
    is based on https://docs.google.com/spreadsheets/d/1X1HTxkI6SqsdpNSkSSivMzpxNT-oeTbjFFDdEkXD30o/edit#gid=695409533

    Download the file and use its location at path2.

    path1 is a textfile containing the credentials for the Reddit API (see
    dummycredits.ini for an example)


    :param path1: (str) Path to reddit credentials
    :param path2: (str) Path to whisky reviews database
    :return: two files: 'reviews.csv' and 'phony_urls.csv'
    """
    creds = import_credentials(path1)
    reddit = praw.Reddit(client_id=creds['client_id'],
                         client_secret=creds['client_secret'],
                         password=creds['password'],
                         username=creds['username'],
                         user_agent=creds['user_agent'])

    keepcols = ['Whisky Name', 'Link To Reddit Review']
    df = pd.read_csv(path2, usecols=keepcols, sep=';')
    df.columns = ['name', 'url']
    df.drop_duplicates('url', inplace=True)

    for n, (name, url) in df.iterrows():
        try:
            subm = reddit.submission(url=url)
            rvw = re.sub('\*', '', subm.comments[0].body)
        except (IndexError, praw.exceptions.ClientException):
            print('could not fetch #', n, ': ', name)
            with open('phony_urls.csv', 'a') as r:
                wr = csv.writer(r)
                url_info = [n, name, url]
                wr.writerow(url_info)
                continue
        except:
            print('something went wrong.. will just skip')
            continue

        lst = re.findall('([A-Z][a-z]+?: .*)', rvw)
        lst.insert(0, name)
        lst.append(url)
        with open('reviews.csv', 'a') as r:
            wr = csv.writer(r)
            wr.writerow(lst)
            print('Processed #', n, ': ', name)
    pass


def import_credentials(loc):
    cred = []
    with open(loc, 'r') as r:
        wr = csv.reader(r)
        for line in wr:
            cred.extend(line)
    creds = dict([re.split('=', x) for x in cred])
    return creds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-doc1', dest='credentials', required=True,
                        help='path of Reddit API credentials, like in '
                             'dumyycredits.ini')
    parser.add_argument('-doc2', dest='review_urls', required=True,
                        help='path to the base document containing whisky name'
                             'and reddit urls.')
    args = parser.parse_args()

    scrape(args.credentials, args.review_urls)
    pass


if __name__ == '__main__':
    main()
