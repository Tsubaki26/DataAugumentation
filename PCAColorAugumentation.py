import cv2
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as LA


def pcaDataAugumentation(img):
    """
    PCAを用いて画像データ増強を行う関数
    """

    """
    画像の正規化
    """
    # 2次元配列に変換
    image_reshape = img.reshape(img.shape[0] * img.shape[1], 3).astype(np.float32)
    # 正規化
    image_reshape = (image_reshape - np.mean(image_reshape, axis=0)) / np.std(
        image_reshape, axis=0
    )
    # 分散共分散行列の計算
    Sigma = np.cov(image_reshape, rowvar=False)
    print("Sigma:\n", Sigma)

    """
    固有値，固有ベクトルの計算
    """
    # 固有値・固有ベクトルの計算（0: 3x1固有値，1: 3x3固有ベクトル）
    results = LA.eig(Sigma)
    eigen_val = results[0]
    eigen_vec = results[1]
    print("Eigenvalue:\n", eigen_val)
    print("EigenVector:\n", eigen_vec)

    """
    画像加工
    """
    # 平均 0,標準偏差 0.1の正規分布に従う乱数を生成
    alpha = np.random.normal(0, 0.1, 3)
    print(f"alpha: {alpha}")

    # 画像に加算するベクトルの計算
    add_vec = np.dot(eigen_vec, alpha * eigen_val) * 255.0
    print(add_vec)
    add_vec = add_vec.astype(np.int32)[np.newaxis, np.newaxis, :]

    # 0-255の制限付きで加算
    image_processed = np.clip(img + add_vec, 0, 255)
    image_processed = image_processed.astype(np.uint8)
    image_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)

    # RGB画像を返す
    return image_processed


def showHist(img_array):
    """
    ヒストグラムの計算・可視化を行う関数
    """
    colors = ("b", "g", "r")

    num_img = len(img_array)
    fig, ax = plt.subplots(2, num_img, figsize=(15, 5))
    hist_array = np.zeros((256, 3))

    # 画像表示
    for i in range(num_img):
        if i == 0:
            ax[0, 0].set_title("original")
        ax[0, i].imshow(img_array[i])

    # ヒストグラム表示
    for i in range(num_img):
        for j, color in enumerate(colors):
            hist = cv2.calcHist([img_array[i]], [j], None, [256], [0, 256])

            # squeeze()で(256,1)->(256,)に変換．
            hist_array[:, j] = hist.squeeze()

            ax[1, i].plot(hist, color=color)

    fig.tight_layout()

    plt.show()


def main():
    img_path = "./img/train_1.jpg"
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("image shape: ", img.shape)

    num_pr = 3
    img_array = []
    img_array.append(img_RGB)

    # データ増強
    for i in range(num_pr):
        img_array.append(pcaDataAugumentation(img))

    showHist(img_array)


if __name__ == "__main__":
    main()
