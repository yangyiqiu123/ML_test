from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from tensorflow import keras

basedir = Path(__file__).parent.parent
dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    return "hello"


@dt.route("/detected")
def detected():
    return "a"


def load_image(filename, reshape_size=(32, 32)):
    # 加載圖片

    dir_image = str(basedir / "data" / "original" / filename)
    # 建立圖片資料的物件
    image_obj = PIL.Image.open(dir_image).convert("RGB")

    # 修改圖片資料的尺寸
    image = image_obj.resize(reshape_size)
    return image, filename


def detect():
    filename = "a.jpg"
    labels = current_app.config["LABELS"]
    image = load_image(filename)
    # 將圖片轉換為 NumPy 陣列
    image_array = np.array(image)

    # 正規化像素值到 [0, 1]
    normalized_image = image_array / 255.0

    # 加載模型
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation="softmax"))

    try:
        model.load_weights("./cifarCnnModel.h5")
        print("success")
    except:
        print("error")

    prediction = model.predict(normalized_image)[0]
    print("predict:", labels[np.argmax(prediction[0])])
