!nvidia-smi

!pip install ultralytics

from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo mode=checks

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Load the image
image_path = '/content/drive/MyDrive/project3/motherboard_image.JPEG'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blurring
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 20, 100)

# Perform a dilation + erosion to close gaps between edge segments
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours in the edged image
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and remove small contours that are not the motherboard
contours = sorted(contours, key=cv2.contourArea, reverse=True)
pcb_contour = contours[0]  # Assuming the largest contour is the PCB

# Create an empty mask and draw the largest contour, which is the PCB
mask = np.zeros_like(gray)
cv2.drawContours(mask, [pcb_contour], -1, (255), thickness=cv2.FILLED)

# Bitwise-AND to extract the motherboard area
extracted = cv2.bitwise_and(image, image, mask=mask)


# Save the result
output_path = '/content/drive/MyDrive/Project3/extracted_motherboardfinal.png'
cv2.imwrite(output_path, extracted)


plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
plt.title('Extracted Motherboard')
plt.show()

print(output_path)


!yolo task=detect mode=train model=yolov8m.pt data = /content/drive/MyDrive/project3/data/data.yaml epochs = 180 imgsz = 900 batch =15  name = ttruehighmodell

!pip install pillow

from PIL import Image
model= YOLO('/content/runs/detect/ttruehighmodell/weights/best.pt')

# Run batched inference on a list of images
results = model('/content/drive/MyDrive/project3/data/evaluation/rasppi.jpg')  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results:
    im_array = r.plot(line_width=5, font_size=5)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/project3/raspihigh.jpg')  # save image


results2 = model('/content/drive/MyDrive/project3/data/evaluation/arduno.jpg')

# Process results generator
for result in results2:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results2:
    im_array = r.plot(line_width=2, font_size=2)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/project3/ardunohigh.jpg')  # save image

results3 = model('/content/drive/MyDrive/project3/data/evaluation/ardmega.jpg')

# Process results generator
for result in results3:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results3:
    im_array = r.plot(line_width=5, font_size=5)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/project3/ardmehigh.jpg')  # save image



