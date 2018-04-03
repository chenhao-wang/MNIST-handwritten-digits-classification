import glob
import cv2
from utils import batchnorm_forward, predict
import numpy as np

# load handwritten digit images from test-images folder
filenames = [img for img in glob.glob("test-images/*")]
filenames.sort()
images = []
names = []
for img in filenames:
    names.append(str(img))
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img_gray, dsize=(28, 28))
    img = 255 - img
    images.append(img)

images = np.reshape(images, (len(images), 28 * 28))
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

weights = np.load('weights/weights.npy')
parameters = weights.item()

if __name__ == '__main__':
    images_norm, _ = batchnorm_forward(images)
    _, pred = predict(parameters, images_norm, labels)
    # print image name and corresponding predict value
    for i in range(len(names)):
        print(names[i], " predcit = ", pred[i])
