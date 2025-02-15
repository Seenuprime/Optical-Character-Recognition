import tensorflow as tf
import numpy as np
import cv2 as cv

image = cv.imread('detect.png')

def get_contours(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured_image = cv.GaussianBlur(gray_image, (5, 5), 1)
    _, binary_image = cv.threshold(blured_image, 50, 255, cv.THRESH_BINARY)
    edges = cv.Canny(binary_image, 50, 150)
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
    pass