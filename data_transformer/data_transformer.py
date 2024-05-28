import re
import nltk
import pandas as pd
from flask import Flask, jsonify, request, make_response
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from os import environ


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)


@app.route('/transform', methods=['POST'])
def transform():
    request_data = request.get_json()
    data_to_transform = pd.read_json(request_data['data'])
    column_to_transform = request_data['column']

    app.logger.info("Transforming data started.")

    app.logger.info("Removing tags ...")
    data_to_transform[column_to_transform] = data_to_transform[column_to_transform].apply(remove_tags)
    app.logger.info("Removing special chars ...")
    data_to_transform[column_to_transform] = data_to_transform[column_to_transform].apply(special_char)
    app.logger.info("Converting to lower case ...")
    data_to_transform[column_to_transform] = data_to_transform[column_to_transform].apply(convert_lower)
    app.logger.info("Removing stopwords ...")
    data_to_transform[column_to_transform] = data_to_transform[column_to_transform].apply(remove_stopwords)
    app.logger.info("Lemmatizing words ...")
    data_to_transform[column_to_transform] = data_to_transform[column_to_transform].apply(lemmatize_word)

    app.logger.info("Transforming data finished.")

    return make_response(jsonify({"data": data_to_transform.to_json()}), 200)


def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)


def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews


def convert_lower(text):
    return text.lower()


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]


def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('DATA_TRANSFORMER_PORT'))
