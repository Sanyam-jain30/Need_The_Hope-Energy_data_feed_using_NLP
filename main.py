# importing libraries
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import warnings

warnings.filterwarnings('ignore')

# cleaning text
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

urls = []
headlines = []
corpus = []


# climate website
def climate_website(url):
    source = requests.get(url).text

    soup = BeautifulSoup(source, "lxml")

    for a in soup.findAll('a', class_='field-group-link card-link'):
        urls.append(urljoin(url, str(a['href'])))

        div = a.find('div', class_='vertical-card-content')

        h4 = div.find('h4', class_='faux-full-title').text
        headlines.append(h4)


url = 'https://climate.mit.edu/explainers'
source = requests.get(url).text

soup = BeautifulSoup(source, "lxml")
ul = soup.find('ul', class_='pager__items js-pager__items')

for li in ul.findAll('li', class_='pager__item'):
    if 'pager__item--next' not in li['class']:
        climate_website(urljoin(url, str(li.a['href'])))

# netl website
url = 'https://netl.doe.gov/carbon-management'
source = requests.get(url).text

soup = BeautifulSoup(source, "lxml")

for div in soup.findAll('div', class_='col-md-11'):
    h4 = div.find('h4')

    if h4.a is not None:
        headlines.append(h4.text)
        urls.append(h4.a['href'])

    ul = div.find('ul')

    for li in ul.findAll('li'):
        if li.a is not None:
            headlines.append(li.text)
            a = li.a['href']
            if a.startswith('http'):
                urls.append(a)
            else:
                urls.append(urljoin(url, a))


# clean data
def clean_data(text):
    content = re.sub('[^a-zA-Z]', ' ', text)
    content = content.lower()
    content = content.split()

    ps = PorterStemmer()
    content = [ps.stem(word) for word in content if not word in set(stopwords.words('english') + ['u', 'r'])]
    content = ' '.join(content)
    return content


# fetching the text from the identified urls
def get_summary(urls_given):
    for url in urls_given:
        source = requests.get(url).text

        soup = BeautifulSoup(source, "lxml")

        if url.__contains__('climate.mit.edu'):
            div = soup.find('div', class_='left-column')
            p = div.find('p')
            corpus.append(clean_data(p.text))
        elif url.__contains__('netl.doe.gov'):
            div = soup.find('div', class_='node')
            div2 = div.find_all('div')[1]
            for p in div2.find_all('p'):
                if p.text == '' or p.text == ' ' or len(p.text) == 1 or p.strong != None:
                    continue
                else:
                    corpus.append(clean_data(p.text))
                    break


import pandas as pd

# Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(headlines).toarray()
df = pd.DataFrame(data=X, columns=vectorizer.get_feature_names_out())

# LSA (Latent semantic analysis)
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
df = lsa.fit_transform(df)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

# Checking for optimal number of clusters for KMeans - Elbow method
from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(2, 20)
for k in K:
    km = KMeans(n_clusters=k, random_state=1, max_iter=100)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)

import matplotlib.pyplot as plt

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Result:
# kMeans - We got the best number of clusters to be 4, 9, 16 from the plot (the point where it bends)

km4 = KMeans(n_clusters=4, random_state=1, max_iter=100)
km4 = km4.fit(df)

df = pd.DataFrame(df)
df['Cluster'] = km4.labels_.astype(int)
df['url'] = urls

feed = input("Enter the feed: ")
feed = clean_data(feed)
feed = [feed]

feed = vectorizer.transform(feed).toarray()
y = pd.DataFrame(data=feed, columns=vectorizer.get_feature_names_out())

y = lsa.transform(y)

result = km4.predict(y)

urlsFound = []
print('URLs having similar content as feed content: ')
for row in range(len(df)):
    if (df.loc[row, "Cluster"] == result):
        print(df.loc[row, "url"])
        urlsFound.append(df.loc[row, "url"])

print('\nFetching content from the urls to generate summary')
get_summary(urlsFound)

word_frequencies = {}
for sentence in corpus:
    for word in sentence.split(' '):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

sentence_scores = {}
for sentence in corpus:
    for word in sentence.split(' '):
        if word in word_frequencies.keys():
            if sentence not in sentence_scores.keys():
                sentence_scores[sentence] = word_frequencies[word]
            else:
                sentence_scores[sentence] += word_frequencies[word]

import heapq

number_of_sentences = int(sum(sentence_scores.values()) / len(sentence_scores))
summary_sentences = heapq.nlargest(number_of_sentences, sentence_scores, key=sentence_scores.get)

# We got the summary
summary = ' '.join(summary_sentences)
print("Before paragraph phrasing the summary", summary)

# To phrase the summary and make it more informative for user
# It needs to download some files around 2.8GB size - may take some time
from transformers import *

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=1, num_beams=5):
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


phrase_summary_sentences = []
for sentence in summary_sentences:
    phrase_sentence = get_paraphrased_sentences(model, tokenizer, sentence)
    phrase_summary_sentences.append(phrase_sentence[0])

print(phrase_summary_sentences)

from textblob import TextBlob

correct_phrase_summary_sentences = []
for phrase_sentence in phrase_summary_sentences:
    phrase_sentence = TextBlob(phrase_sentence)
    correct_phrase_summary_sentences.append(phrase_sentence.correct())

final_summary = ' '.join(map(str, correct_phrase_summary_sentences))
print("The final summary is: ")
print(final_summary)

"""
dataset = pd.DataFrame({
    'headline': headlines,
    'url': urls,
    'corpus': corpus
})

dataset.to_csv('fetched.csv')
"""
