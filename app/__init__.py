import os

from flask import Flask


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/test')
    def test():
        return 'Hello, World'

    from app import home, documents, about, contact
    app.register_blueprint(home.bp)
    app.register_blueprint(documents.bp)
    app.register_blueprint(about.bp)
    app.register_blueprint(contact.bp)

    return app
