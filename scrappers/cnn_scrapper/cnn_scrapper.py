from flask import Flask, jsonify, make_response
import pandas as pd
import requests
from bs4 import BeautifulSoup
from os import environ

app = Flask(__name__)

website_url = "https://edition.cnn.com"

# key - category in db, value - category in website url
categories = {
    "politics": "politics",
    "business": "business",
    "entertainment": "entertainment",
    "sport": "sport",
    "tech": "business/tech"
}

def scrape_urls(label):
    try:
        app.logger.info(f"Scrapping urls for {label} news ...")
        request = requests.get(website_url + "/" + label)
        recipe_soup = BeautifulSoup(request.content, 'html.parser')
        news_urls = [a['href'] if a['href'].startswith('http') else website_url + a['href'] for a in recipe_soup.findAll('a', attrs={"class": "container__link"})]
        app.logger.info(f"Successfully scrapped {len(news_urls)} urls for {label} news ...")
        return news_urls
    except Exception as e:
        print("Failed to extract data for " + label)


def scrape_article(url):
    try:
        request = requests.get(url)
        recipe_soup = BeautifulSoup(request.content, 'html.parser')
        if recipe_soup:
            article_text = recipe_soup.find("div", attrs={"class": "article__content"})
            if article_text:
                paragraph = article_text.find("p", attrs={"class": "paragraph"})
                return paragraph.text
    except Exception as e:
        print("Failed to extract " + url)
        print(e)


@app.route('/extract_news', methods=['GET'])
def extract_news():
    app.logger.info("Scrapping news ...")
    data = []
    for category in categories.keys():
        data.extend([{"text": scrape_article(url), "label": category} for url in scrape_urls(categories[category])])

    app.logger.info(f"Successfully scrapped {len(data)} news ...")

    df = pd.DataFrame.from_dict(data)
    df = df.dropna()
    df_json = df.to_json()

    return make_response(jsonify({"data": df_json}), 200)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('CNN_SCRAPPER_PORT'))

