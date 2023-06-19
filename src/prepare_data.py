import os
from skimage import io


def dichrop(img, height, width):
    if img.shape[0] <= height and img.shape[1] <= width:
        print(img.shape)
        return img
    else:
        print("CROP")
        imgs = [img]
        if img.shape[0] > height:
            imgs = [img[: img.shape[0] // 2, :, :], img[img.shape[0] // 2:, :, :]]
        if img.shape[1] > width:
            imgs = [subimg[:, : subimg.shape[1] // 2, :] for subimg in imgs] + [
                subimg[:, subimg.shape[1] // 2:, :] for subimg in imgs
            ]

        return [dichrop(img, height, width) for img in imgs]


save_path = "/home/qosu/data/munch_paintings/usable"

data_path = "/home/qosu/data/munch_paintings/raw"
extensions = [".png", ".jpg", ".jpeg"]
paths = [
    os.path.join(data_path, p)
    for p in os.listdir(data_path)
    if os.path.splitext(p)[1] in extensions
]

width = 512
height = 512

for p in paths:
    img = io.imread(p)
    if len(img.shape) > 2:
        imgs = dichrop(img, height, width)
        print(type(imgs[0]), img.shape)
        input()
