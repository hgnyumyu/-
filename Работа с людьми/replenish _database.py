import cv2
import numpy as np 
    
prototxt_path = ".idea/deploy.prototxt.txt"
model_path = ".idea/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
User = input("Введите ФИО: ")
capt = cv2.VideoCapture(0)
while True:
    reg, image = capt.read()
    image = cv2.flip(image, 1)
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # устанавливаем на вход нейронной сети изображение
    model.setInput(blob)
    # выполняем логический вывод и получаем результат
    output = np.squeeze(model.forward())
    font_scale = 1.0
    for i in range(0, output.shape[0]):
        # получить уверенность
        confidence = output[i, 2]
        # если достоверность выше 50%, то нарисуйте окружающий прямоугольник
        if confidence > 0.5:
            cv2.imwrite(f"./staff/{User}.jpg", image)
            # получить координаты окружающего блока и масштабировать их до исходного изображения
            box = output[i, 3:7] * np.array([w, h, w, h])
            # преобразовать в целые числа
            start_x, start_y, end_x, end_y = box.astype(int)
            # рисуем прямоугольник вокруг лица
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(0, 255, 0), thickness=2)
            break
    break    
    cv2.imshow("testing_face_recognition_in_a_video_stream_#158", image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capt.release()
cv2.destroyAllWindows()
