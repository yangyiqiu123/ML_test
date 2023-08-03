from pathlib import Path

from apps.config import config
from flask import Flask, render_template


def create_app(config_key):
    app = Flask(__name__)

    app.config.from_object(config[config_key])

    from apps.detector import views as dt_views

    app.register_blueprint(dt_views.dt)

    return app
