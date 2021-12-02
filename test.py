import os
import numpy as np
import cv2
import PIL.Image as Image

import Train

npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaFlattenedImages = npaFlattenedImages.reshape((int(npaFlattenedImages.size / 20), 20))

print(npaFlattenedImages[30:60, 0:20])
cv2.imwrite("temp.jpg", npaFlattenedImages[0:30, 0:20])
# image = npaFlattenedImages.reshape((int(npaFlattenedImages.size / 200), 200))
# # print(image[0:30, 0:20])
# for i in range(18):
#    for j in range(10):
#       image[i * 20, i * 20 + 20][j * 30,j * 30 + 30] = npaFlattenedImages[i * 10 + j:i * 10 + j + 30, 0:20]
# print(image)

# cv2.imshow("input", image[0:30, 0:20])
# cv2.waitKey(0)
# print(npaFlattenedImages.size)
# print(type(npaFlattenedImages))
# print(len(npaFlattenedImages))
# print(len(npaFlattenedImages[0]))
# 定义图像拼接函数

IMAGES_PATH = r'D:\pics22223\\'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_SIZE = 1000  # 每张小图片的大小
IMAGE_ROW = 18
IMAGE_COLUMN = 10
IMAGE_SAVE_PATH = r'd:\gisoracle.jpg'  # 图片转换后的地址

def image_compose():
   to_image = Image.new('L', (IMAGE_COLUMN * 20, IMAGE_ROW * 30), 256)  # 创建一个新图
   # 循环遍历，把每张图片按顺序粘贴到对应位置上
   for y in range(0, IMAGE_ROW):
      for x in range(0, IMAGE_COLUMN):
         print(x, y)
         cv2.imwrite("temp.jpg", npaFlattenedImages[(y * 10 + x) * 30:(y * 10 + x) * 30 + 30, 0:20])
         from_image = Image.open("temp.jpg")
         to_image.paste(from_image, (x * 20, y * 30, x * 20 + 20, y * 30 + 30))
         to_image.save("temp2.jpg")
   to_image.show("input")  # 保存新图


image_compose()  # 调用函数