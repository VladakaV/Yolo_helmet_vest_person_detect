# import numpy as np
# import pandas as pd
# import os
#
# import yaml
#
# dict_file = {'train':'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/images/train' ,
#             'val': 'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/images/val',
#             'nc' : 3,
#             'names' : ['helmet','vest','head']}
#
# with open('C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/doc/hard_head.yaml', 'w+') as file:
#     documents = yaml.dump(dict_file,file)
#
# from ultralytics import YOLO
#
# model = YOLO("yolov8s.pt")
#
# model.train(data="C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/doc/hard_head.yaml", epochs=20)
# from cProfile import label

#photo
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# from ultralytics import YOLO
#
# # Инициализация модели
# model = YOLO("C:/Users/vlada/Desktop/best.pt")
#
#
# def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#     lw = max(round(sum(image.shape) / 2 * 0.003), 2)
#     p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#     cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(lw - 1, 1)  # font thickness
#         w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
#         outside = p1[1] - h >= 3
#         p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#         cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(image,
#                     label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
#                     0,
#                     lw / 3,
#                     txt_color,
#                     thickness=tf,
#                     lineType=cv2.LINE_AA)
#
#
# def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
#     # Labels and colors initialization
#     if labels == []:
#         labels = {0: u'__background__', 1: u'helmet', 2: u'vest', 3: u'head'}
#     if colors == []:
#         colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24)]
#
#     for box in boxes:
#         # Adding score in label if needed
#         if score:
#             label = labels.get(int(box[-1]) + 1, 'Unknown') + " " + str(round(100 * float(box[-2]), 1)) + "%"
#         else:
#             label = labels.get(int(box[-1]) + 1, 'Unknown')
#
#         if conf:
#             if box[-2] > conf:
#                 color = colors[int(box[-1])]
#                 box_label(image, box, label, color)
#         else:
#             color = colors[int(box[-1])]
#             box_label(image, box, label, color)
#
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image
#
#
# def plot_results(results, image_path = None, image_data = None, result_name = None, box_type = False, labels=[]):
#     image = None
#     if image_path:
#         image = Image.open(image_path)
#     else:
#         image = image_data
#
#     image = np.asarray(image).copy()  # Ensure the image is writable by creating a copy
#
#     img = None
#     if not box_type:
#         # img = plot_bboxes(image, results[0].boxes.boxes, labels=labels, score=True)
#         img = plot_bboxes(image, results[0].boxes.data, labels=labels, score=True)
#     else:
#         img = plot_bboxes(image, results, score=True)
#
#     result_image_name = result_name if result_name else image_path.split('/')[-1]
#     result_path = 'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/results/' + result_image_name
#     cv2.imwrite(result_path, img)
#
#     # Display image with matplotlib
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')  # Remove axes
#     plt.show()
#
#
# import IPython.display
#
# test_images = os.listdir(
#     'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/images/test/')
#
#
# def predict(test_images, index):
#     labels = {0: u'__background__', 1: u'helmet', 2: u'vest', 3: u'head'}
#     path = 'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/images/test/' + \
#            test_images[index]
#     results = model(path)
#     result_name = 'hardhat_pred_' + test_images[index] + '.jpg'
#     plot_results(results, image_path=path, labels=labels, result_name=result_name)
#     return IPython.display.Image(
#         "C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/results/" + result_name)
#
## predict(test_images, 1)

#VIDEO
import cv2
import numpy as np
from ultralytics import YOLO

# model_person = YOLO("C:/Users/vlada/Desktop/yolo_person.pt")
model_person = YOLO("yolov8s.pt")
model = YOLO("C:/Users/vlada/Desktop/best.pt")


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # тут при заданных параметрах текста типа смотрим сколько он будет занимать по высоте и ширине, чтобы потом корректно разместить
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None): #conf - порог процента уверенности, чтобы объект отобразился
    # Labels and colors initialization
    if labels == []:
        labels = {0: u'__background__', 1: u'helmet', 2: u'vest', 3: u'head'}
    if colors == []:
        colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24)]

    for box in boxes:
        # Adding score in label if needed
        if score:
            label = labels.get(int(box[-1]) + 1, 'Unknown') + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label =''

        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def process_video(input_video_path, output_video_path, labels=[]):
    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()  # Чтение одного кадра
        if not ret:
            break

        # Обнаружение людей на кадре
        result_person = model_person.predict(frame)

        img = None

        print(result_person[0].boxes.data)

        boxes_of_people = []
        boxes_of_helmets = []

        for detection in result_person[0].boxes.data:
            x_min, y_min, x_max, y_max, confidence, class_id = detection

            if int(class_id) == 0:  # Класс 0 — человек
                boxes_of_people.append(detection)
                #если человек, то в его пределах ищет шлем и жилет
                # Извлечение координат и других параметров
                x_min, y_min, x_max, y_max, confidence, class_id = detection.tolist()

                # Приведение координат к целым числам
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                # Создание маски и обрезка изображения
                mask = np.zeros_like(frame)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
                darkened_frame = cv2.bitwise_and(frame, mask)

                # Обрезка изображения человека
                person_roi = darkened_frame[y_min:y_max, x_min:x_max]

                # Применить модель поиска каски и жилета на области
                results = model(person_roi)

                is_helmet_detected = False
                is_vest_detected = False

                print(model.names)

                # Отображение результатов на оригинальном кадре
                for detection in results[0].boxes.data:
                    h_x_min, h_y_min, h_x_max, h_y_max, confidence, class_id = detection

                    # Преобразовать координаты обратно в систему координат исходного кадра
                    abs_x_min = x_min + int(h_x_min)
                    abs_y_min = y_min + int(h_y_min)
                    abs_x_max = x_min + int(h_x_max)
                    abs_y_max = y_min + int(h_y_max)

                    # Добавить bounding box для каски или жилета на оригинальный кадр
                    box_info = [abs_x_min, abs_y_min, abs_x_max, abs_y_max, confidence, class_id]
                    boxes_of_helmets.append(box_info)

                    if (class_id == 0):
                        is_helmet_detected = True
                    elif (class_id == 1):
                        is_vest_detected = True

                if (not is_helmet_detected):
                    print("Шлем не был обнаружен")
                if (not is_vest_detected):
                    print("Жилет не был обнаружен")

        img = plot_bboxes(frame, boxes_of_people, labels=[], score=False) #обводим всех людей на изображении
        img = plot_bboxes(frame, boxes_of_helmets, labels=labels, score=True) #обводим все каски не изображении

        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        cv2.imshow("Frame", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


input_video_path = 'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/videos/test_video.mp4'
output_video_path = 'C:/Users/vlada/.cache/kagglehub/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/versions/2/results/output_video.mp4'

labels = {0: u'__background__', 1: u'helmet', 2: u'vest', 3: u'head'}
process_video(input_video_path, output_video_path, labels)











