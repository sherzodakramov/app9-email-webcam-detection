import glob
import cv2
import time
import os
from emailing import send_email
from threading import Thread


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# We give a second to camera to start/load video for avoiding black frames
time.sleep(1)

first_frame = None
status_list = []
count = 1


def clean_folder():
    print("clean_folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean_folder function ended")


while True:
    status = 0
    check, frame = video.read()
    # Provide the gray frame (therefore comparing between frames will be easier)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Make little blurred frame, so we don't need that all precisions,
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (15, 15), 0)

    # We hold the first frame and store it to the variable
    if first_frame is None:
        first_frame = gray_frame_gau

    # And compare the first frame with current frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # We will classify the pixels based on a threshold
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

    # Remove the noise from threshold
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    # cv2.imshow("My video", dil_frame)

    # Find the contours
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        email_thread.start()

    print(status_list)

    cv2.imshow("Video", frame)

    # Create keyboard key object
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

clean_thread.start()
video.release()


