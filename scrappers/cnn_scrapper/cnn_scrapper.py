from flask import Flask, jsonify, request
import pandas as pd
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)


def scrape_urls(url_label, label):
    request = requests.get("https://edition.cnn.com/" + label)
    recipe_soup = BeautifulSoup(request.content, 'html.parser')
    news_urls = ["https://edition.cnn.com" + a['href'] for a in recipe_soup.findAll('a', attrs={"class": "container__link"})]
    return news_urls


def scrape_article(url):
    request = requests.get(url)
    recipe_soup = BeautifulSoup(request.content, 'html.parser')
    article_text = recipe_soup.find("div", attrs={"class": "article__content"})
    article_text.find("p", attrs={"class": "paragraph"})
    return article_text.text


@app.route('/extract_news')
def extract_news():
    categories = ["politics", "business", "entertainment", "sport", "business/tech"]

    data = []
    for category in categories:
        data.extend([{"text": scrape_article(url), "category": category}
                     if category != "business/tech"
                     else {"text": scrape_article(url), "category": "tech"}
                     for url in scrape_urls(category)])

    response = {'prediction': prediction[0]}

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
