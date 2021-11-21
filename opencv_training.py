import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../training.jpg')

## problem 2 color detection: HSV + inRange() 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
# define range of red color in HSV 
lower_red = np.array([156,43,46])
upper_red = np.array([180,255,255])

mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_red = cv2.inRange(hsv, lower_red, upper_red)
#mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)


red = np.zeros(img.shape, np.uint8)
red[:,:,2] = red[:,:,2]+255
blue = np.zeros(img.shape, np.uint8)
blue[:,:,0] = blue[:,:,0]+255

p2 = cv2.bitwise_and(red, red, mask = mask_red) + cv2.bitwise_and(blue, blue, mask = mask_blue)


## problem 1 houghcircles and problem 3
p1 = np.copy(img)
p3 = np.copy(img)

# find circles from blue mask
circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT, 1.5, 2000, param1=100, param2=25, minRadius=250, maxRadius=500)
circles = np.uint16(np.around(circles))
num_blue = circles.shape[1]
for idx, c in enumerate(circles[0,:]):
    # draw circle
    cv2.circle(p1, (c[0],c[1]), c[2], (50,205,50), 20)
    cv2.circle(p1, (c[0],c[1]), 2, (50,205,50), 20)
    # index
    cv2.putText(p3, str(idx+1), (c[0]-100,c[1]+100), cv2.FONT_HERSHEY_TRIPLEX, 10, (50,205,50), 10, cv2.LINE_AA)

# find circles from red mask
circles = cv2.HoughCircles(mask_red, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=20, minRadius=250, maxRadius=500)
circles = np.uint16(np.around(circles))
num_red = circles.shape[1]
for idx, c in enumerate(circles[0,:]):
    # draw circle
    cv2.circle(p1, (c[0],c[1]), c[2], (50,205,50), 20)
    cv2.circle(p1, (c[0],c[1]), 2, (50,205,50), 20)
    # index
    cv2.putText(p3, str(idx+1), (c[0]-100,c[1]+100), cv2.FONT_HERSHEY_TRIPLEX, 10, (50,205,50), 10, cv2.LINE_AA)



plt.subplot(221)
plt.title('original image')
plt.axis('off')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.title('problem 1')
plt.axis('off')
plt.imshow(cv2.cvtColor(p1, cv2.COLOR_BGR2RGB))
plt.subplot(223)
plt.title('problem 2')
plt.axis('off')
plt.imshow(cv2.cvtColor(p2, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.title('problem 3')
plt.axis('off')
plt.imshow(cv2.cvtColor(p3, cv2.COLOR_BGR2RGB))
plt.show()


