import cv2
import numpy as np
import pytesseract
# создать новый объект камеру
capt = cv2.VideoCapture('./test/test_video.mp4')
# инициализировать поиск лица (по умолчанию каскад Хаара)
car_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
config = r' --oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
path = r'./save_img/'
with open("numbers.txt") as f:
    nums_list = f.readlines()

for i in range(len(nums_list)):
    nums_list[i]= nums_list[i].strip()
nums_in = []
while capt.isOpened():
    # чтение изображения с камеры
    _, image = capt.read()
    #image = cv2.flip(image, 1)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 70, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    car_numbers = car_cascade.detectMultiScale(image_gray, 1.3, 5)

    for x, y, width, height in car_numbers:
        image_car_number = image_gray[y: (y + height), x: (x + width)].copy()
        #number = pytesseract.image_to_string(image_car_number, config=config).strip()
        print(number)
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
        #cv2.putText(image, number, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        if number in nums_list and not(number in nums_in):
            print("Доступ разрешен")
            nums_in.append(number)
    
    cv2.imshow("image", image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capt.release()
cv2.destroyAllWindows()