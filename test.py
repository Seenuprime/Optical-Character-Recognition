import numpy as np
import cv2 as cv
import tensorflow as tf

model = tf.keras.models.load_model("digit_char_model.keras") 

class_names = ['#', '$', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load image
image = cv.imread("detect1.png", 0)

print("Image shape:", image.shape)

# Blur to reduce noise
blured_image = cv.GaussianBlur(image, (5, 5), 1)

# Convert to binary image
_, binary_image = cv.threshold(blured_image, 50, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

cv.imshow("Binary Image", binary_image)
cv.waitKey(0)

# Edge detection
edges = cv.Canny(binary_image, 10, 20)

# cv.imshow("Edges", edges)
# cv.waitKey(0)

# Find contours
contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

min_width = 30
min_height = 30

bbox = [cv.boundingRect(contour) for contour in contours if cv.boundingRect(contour)[2]>min_width and cv.boundingRect(contour)[3]>min_height] 
bbox = list(set(bbox))
bbox.sort(key=lambda b: (b[1] // b[3], b[0])) 

for (x, y, w, h) in bbox:
    # print(x, y, w, h)
    # cv.rectangle(image, (x-5, y-5), (x+w+5, y+h+5), color=(0, 255, 0), thickness=3)
    cut_image = image[y:y+h, x:x+w]
    padding = ((5, 5), (5, 5))
    padded_image = np.pad(cut_image, padding, mode='maximum')

    thresh = cv.threshold(padded_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    thresh = cv.resize(thresh, (32, 32), interpolation=cv.INTER_CUBIC)

    final_image = np.expand_dims(thresh, axis=-1)
    final_image = tf.expand_dims(final_image, axis=0)
    predicted = model.predict(final_image).argmax()
    pred_class = class_names[predicted]
    print(pred_class)

    cv.imshow(pred_class, thresh)
    if cv.waitKey(0) & 0xFF == ord("q"):
        break
    cv.destroyAllWindows()

cv.destroyAllWindows()

# cv.imshow("Contours", blured_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
