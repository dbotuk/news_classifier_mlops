import pandas as pd
from flask import Flask, jsonify, request, make_response
from flask_sqlalchemy import SQLAlchemy
from os import environ

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
db = SQLAlchemy(app)


class News(db.Model):
    __tablename__ = 'news'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(), nullable=False)
    label = db.Column(db.String(), nullable=False)

    def json(self):
        return {'id': self.id, 'text': self.text, 'label': self.label}


with app.app_context():
    db.create_all()


@app.route('/add_all_news', methods=['POST'])
def add_all_news():
    try:
        request_data = request.get_json()
        data = pd.read_json(request_data['data'])
        data.to_sql('news', db.engine, if_exists='append', index=False)
        return make_response(jsonify({"message": "News added successfully"}), 201)
    except Exception as e:
        return make_response(jsonify({"message": "Error during adding news"}), 500)


@app.route('/add_news', methods=['POST'])
def add_news():
    try:
        data = request.get_json()
        new_news = News(text=data['text'], label=data['label'])
        db.session.add(new_news)
        db.session.commit()
        return make_response(jsonify({"message": "News added successfully"}), 201)
    except Exception as e:
        return make_response(jsonify({"message": "Error during adding news"}), 500)


@app.route('/news', methods=['GET'])
def get_all_news():
    try:
        news_list = News.query.all()
        return make_response(jsonify({"news": [news.json() for news in news_list]}), 200)
    except Exception as e:
        return make_response(jsonify({"message": "error getting news"}), 500)


@app.route('/news/<int:id>', methods=['GET'])
def get_news(id):
    try:
        news = News.query.filter_by(id=id).first()
        if news:
            return make_response(jsonify({"news": news.json()}), 200)
        return make_response(jsonify({"message": "news not found"}), 404)
    except Exception as e:
        return make_response(jsonify({"message": "error getting news"}), 500)


@app.route('/update_news/<int:id>', methods=['POST'])
def update_news(id):
    try:
        news = News.query.filter_by(id=id).first()
        if news:
            data = request.get_json()
            news.text = data['text']
            news.label = data['label']
            db.session.commit()
            return make_response(jsonify({"message": "news updated"}), 200)
        return make_response(jsonify({"message": "news not found"}), 404)
    except Exception as e:
        return make_response(jsonify({"message": "error updating news"}), 500)


@app.route('/delete_news/<int:id>', methods=['POST'])
def delete_news(id):
    try:
        news = News.query.filter_by(id=id).first()
        if news:
            db.session.delete(news)
            db.session.commit()
            return make_response(jsonify({"message": "news deleted"}), 200)
        return make_response(jsonify({"message": "news not found"}), 404)
    except Exception as e:
        return make_response(jsonify({"message": "error deleting news"}), 500)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get('DB_SERVER_PORT'))
