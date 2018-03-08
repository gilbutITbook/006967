# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys
from skimage.feature import hog
from skimage import data, color, exposure, io


if __name__ == '__main__':
    argvs = sys.argv

    ## 데이터의 읽어 들이기(그레이스케일)
    image = io.imread(argvs[1], as_grey=True)

    ## HOG 특징량을 계산
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                    cells_per_block=(5, 5), visualise=True)

    ## 원래 이미지 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True,
                    sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    ## HOG특징량 그리기

    hog_image_rescaled = exposure.rescale_intensity(hog_image,
                    in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
