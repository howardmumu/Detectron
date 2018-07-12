import cv2

imgpath=r"G:/人_机动车_非机动车/train2017/000036.jpg"
img=cv2.imread(imgpath)
cv2.imshow("1",img)
cv2.waitKey(0)