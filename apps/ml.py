import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

try:
    model = load_model("./Model.h5")
    print("success")
except:
    print("error")


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap="binary")
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap="binary")
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(np.argmax(prediction[idx]))

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# 讀取測試圖片
# img = np.array(Image.open('test.png'))
img = np.array(Image.open("qq.png"))
# img = np.array(Image.open('110321005.jpg'))


# 顯示測試圖片
plot_image(img)

# 建立空的3D Numpy陣列，儲存圖片
data_test = np.empty((1, 3, 32, 32), dtype="uint8")

# 將圖片的RGB通道分離後儲存
data_test[0, :, :, :] = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]

# 將資料轉換為神經網路所需的格式
data_test = data_test.transpose(0, 2, 3, 1)

# 將測試資料進行正規化
data_test_normalize = data_test.astype("float32") / 255.0

# 進行圖片分類預測
prediction = model.predict(data_test_normalize)

# 取出前10個預測結果
prediction = prediction[:10]

# 定義標籤字典
label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# 進行預測機率計算
Predicted_Probability = model.predict(data_test_normalize)


# 定義顯示預測結果與機率的函數
def show_Predicted_Probability(prediction, x_img, Predicted_Probability, i):
    # 顯示預測結果
    print("predict:", label_dict[np.argmax(prediction[i])])
    # 顯示測試圖片
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(data_test[i], (32, 32, 3)))
    # 顯示每個類別的預測機率
    for j in range(10):
        print(label_dict[j] + " Probability:%1.9f" % (Predicted_Probability[i][j]))


# 顯示第一個測試圖片的預測結果與機率
show_Predicted_Probability(prediction, data_test, Predicted_Probability, 0)
