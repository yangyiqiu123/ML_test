from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from flask import Flask, render_template
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

from apps.config import config


def create_app(config_key):
    app = Flask(__name__)

    app.config.from_object(config[config_key])

    from apps.detector import views as dt_views

    app.register_blueprint(dt_views.dt)

    return app
