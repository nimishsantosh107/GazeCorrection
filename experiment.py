import cv2

img = cv2.imread('dataset/NewGazeData/0/chingy-1.jpg')
p1 = (78, 95)
p2 = (176, 96)

print(img.shape)
img = cv2.circle(img, p1, radius=10, color=(0, 255, 0), thickness=-1)
img = cv2.circle(img, p2, radius=10, color=(0, 0, 255), thickness=-1)

# cv2.imshow("label",img)
cv2.imwrite("TEMP.jpg",img)