# EX-5 Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1

Import the required libraries.
### Step2

Convert the image from BGR to RGB.

### Step3

Apply the required filters for the image separately.

### Step4

Plot the original and filtered image by using matplotlib.pyplot.

### Step5

End the program.

## Program:
```
Developed By   : MUKESH P
Register Number: 212222240068
```
### 1. Smoothing Filters

i) Using Averaging Filter
```Python

import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("cad.jpeg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/169
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()




```
ii) Using Weighted Averaging Filter
```Python

kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
iii) Using Gaussian Filter
```Python

gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

```

iv) Using Median Filter
```Python


median=cv2.medianBlur(image2,13)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Blur")
plt.axis("off")
plt.show()

```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()


```
ii) Using Laplacian Operator
```Python


laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()


```

## OUTPUT:
### 1. Smoothing Filters

i) Using Averaging Filter
![313991337-f458aacd-377e-44f7-80a6-595102f3ef02](https://github.com/divakar618/Implementation-of-filter/assets/121932143/572c8745-8032-4b6b-9bfd-19d08e74c2f0)


ii) Using Weighted Averaging Filter
![313991363-98df351b-196a-49ce-9db8-55b48b694181](https://github.com/divakar618/Implementation-of-filter/assets/121932143/78b35a16-b7e5-491f-a623-ebe917c3a959)

iii) Using Gaussian Filter
![313991408-4f15808c-ec5f-4e8e-b36a-7ff37d373029](https://github.com/divakar618/Implementation-of-filter/assets/121932143/dfd403c8-9575-4e94-8579-0a18b4dca5bb)


iv) Using Median Filter
![313991454-22fd1fec-1e5c-4701-86e0-f4d1c516857d](https://github.com/divakar618/Implementation-of-filter/assets/121932143/badcaec6-b190-4c52-a566-8cc9c1b7112d)


### 2. Sharpening Filters

i) Using Laplacian Kernal

![313991498-757802db-014a-47a0-bfc3-660d422f22c8](https://github.com/divakar618/Implementation-of-filter/assets/121932143/91795767-5175-41d8-997f-25101c9a3010)


ii) Using Laplacian Operator

![313991522-21b6646d-215c-4abe-9401-8574e4506a20](https://github.com/divakar618/Implementation-of-filter/assets/121932143/fdd90f56-0826-4594-b722-14e63473ad26)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
