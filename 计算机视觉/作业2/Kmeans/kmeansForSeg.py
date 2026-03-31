# ======1.建立工程并导入sklearn包=======
import numpy as np
import PIL.Image as image  # 加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans  # 加载Kmeans算法



# =======2.加载图片并进行预处理========
def loadData(filePath):
    # 读取图像，并将图像中的pixel归一化到0-1
    img = image.open(filePath)  # 读取图像
    img = img.convert("L")  # 将图像转换为灰度图
    imgData = np.array(img)  # 将图像转换为numpy数组
    imgData = imgData / 255  # 将像素值归一化到0-1之间
    row, col = imgData.shape  # 获取图像的尺寸
    return imgData, row, col  # 以numpy形式返回imgData，以及data尺寸row, col


imgData, row, col = loadData('img1.jpg')  # 加载数据

# 调节此参数
n_clusters_num = 4

# =======—3.加载Kmeans聚类算法========
# 补全代码，使用 K-means方法进行分割预测
imgData = imgData.reshape(row * col, -1)
kmeans = KMeans(n_clusters=n_clusters_num, n_init=10)  # 创建KMeans对象
kmeans.fit(imgData)  # 对图像数据进行聚类
mask = kmeans.predict(imgData)  # 获取聚类结果

# =======4.对像素点进行聚类并输出=======
label = mask.reshape([row, col])
pic_new = image.new("L", (row, col))  # 创建一张新的灰度图保存聚类后的结果
for i in range(row):  # 根据所属类别向图中添加灰度值
    for j in range(col):
        pic_new.putpixel((j, i), int(255 / (label[i][j] + 1)))
pic_new.save("img1_4.jpg", "JPEG")  # 以JPEG格式保存图片
