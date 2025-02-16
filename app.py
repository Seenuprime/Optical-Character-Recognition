import tensorflow as tf
import numpy as np
import cv2 as cv

model = tf.keras.models.load_model("digit_char_model.keras")

class_names = ['#', '$', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def get_contours(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured_image = cv.GaussianBlur(gray_image, (5, 5), 1)
    _, binary_image = cv.threshold(blured_image, 50, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    edges = cv.Canny(binary_image, 10, 20)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    min_width = 30
    min_height = 30
    bbox = [cv.boundingRect(contour) for contour in contours 
            if cv.boundingRect(contour)[2]>min_width and cv.boundingRect(contour)[3]>min_height]
    bbox = list(set(bbox))
    bbox.sort(key=lambda b: (b[1] // b[3], b[0])) 

    return bbox

def final_images(image):
    bbox = get_contours(image)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    cut_images = []
    for (x, y, w, h) in bbox:
        cut_image = gray_image[y:y+h, x:x+w]
        padding = ((5, 5), (5, 5))
        padded_image = np.pad(cut_image, padding, mode='maximum')
        binary_image = cv.threshold(padded_image, 50, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        cut_images.append(binary_image)                  

    return cut_images

def make_pred(model, image, w=32, h=32):
    cut_images = final_images(image)
    
    store_preds = []
    for img in cut_images:
        resized_image = cv.resize(img, (w, h))
        final_image = tf.expand_dims(resized_image, axis=-1)
        final_image = tf.expand_dims(final_image, axis=0)

        pred = model.predict(final_image).argmax()
        store_preds.append(class_names[pred])
    
    return store_preds

image = cv.imread('detect1.png')
preds = make_pred(model, image)

# print(preds)

print("Predicted",''.join(preds))
