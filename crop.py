import cv2

img=cv2.imread("sample.png")
print(img.shape)
h, w = img.shape[:2]
cropped_img = img[int(h*0.45):int(h*0.88), int(w*0.15):int(w*0.45)]
#切り抜きたい部分の座標を入力

cv2.imwrite("crop.png",cropped_img)
cv2.waitKey(0)