import cv2
import numpy as np
import functools

def load_gray_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def load_rgb_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sort the bounding boxes from left to right, top to bottom
# sort by Y first, and then sort by X if Ys are similar
def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ret, thresh  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    _, labels = cv2.connectedComponents(thresh)

    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 70
    upper = total_pixels // 20

    # Loop over the unique components

    for (i, label) in enumerate(np.unique(labels)):
    # If this is the background label, ignore it
        if label == 0:
            continue
        
        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
 
        # If the number of pixels in the component is between lower bound and upper bound, 
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    contours, hierarchy= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

    return boundingBoxes

def segment_characters(image, boundingBoxes):
    for i,bbox in enumerate(boundingBoxes):
        x, y, w, h = bbox

        if x > 60 and y > 60:
            padding = 60
        else:
            padding = 0
        cropped_image = image[y-padding:y+h+padding, x-padding:x+w+padding]
        cv2.imwrite(f'cropped_image_{i}.png', cropped_image)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('segmented_image.png', image)

gray_image = load_gray_image("../images/license2.jpg")
boundingBoxes = preprocess_image(gray_image)
rgb_image = load_rgb_image("../images/license2.jpg")
segment_characters(rgb_image, boundingBoxes)