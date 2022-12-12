import cv2
import numpy as np
import pytesseract
image_file = "2.jpg"
img = cv2.imread(image_file)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
config = r' --oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
def get_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    #pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    #config = r' --oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=2)
    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    values = []
    output = img.copy()
    #print(contours)
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            if w*h > 100:
                image_car_number = img_erode[y: (y + h), x: (x + w)].copy()
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                number = pytesseract.image_to_string(image_car_number, config=config).strip()
                values.append([number, x])

    values.sort(key= lambda x: x[1])
    license_plate = ''
    for value in values:
        license_plate += value[0]
    return license_plate

print(get_license_plate(img))