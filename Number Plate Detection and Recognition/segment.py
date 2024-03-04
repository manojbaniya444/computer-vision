import cv2
import numpy as np
import functools

# from classify import classify_character
from classify_class import ClassificationModel

classification_model = ClassificationModel()  
 
def filter_image(labels,thresh,mask,lower,upper):
    for (i, label) in enumerate(np.unique(labels)):
        # label 0 vaneko background ho
        if label == 0:
            continue

        # otherwise construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # Get the bounding box of the connected component
        (y, x) = np.where(labels == label)
        (topY, topX) = (np.min(y), np.min(x))
        (bottomY, bottomX) = (np.max(y), np.max(x))
        width = bottomX - topX
        height = bottomY - topY


        # filtering the bounding box
        if width > 250 or height > 160:
            continue

        # filtering if the width and height is more than 100px but has less pixel values
        if width > 100 and height > 100 and numPixels < 2000:
            continue
        
        # filtering small squares boxes
        if width < 50 or height < 50:
            if abs(width - height) < 7:
                continue
            
        if topX == 0 or topY == 0:
            continue
        
        if bottomY > 199 or bottomX > 599:
            continue


        # more filtering
        if width > 20 and height > 20:
            # print(width, height, numPixels)
            if numPixels > lower and numPixels < upper:
                # Check if the line starts from the top and ends at more than half the height
                mask = cv2.add(mask, labelMask)
                # print(topX, topY, bottomX, bottomY,width,height,numPixels)
    return mask

def perspective_transform(image, pts):
    # Get the maximum width and height
    max_width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    max_height = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))

    # Set destination points
    dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype='float32')

    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply perspective transform
    result = cv2.warpPerspective(image, matrix, (int(max_width), int(max_height)))

    return result

def multi_color_masking(hsv_image):
    lower1 = np.array([0, 50, 50])  # 50 can be good for capturing low red also
    upper1 = np.array([10, 255, 255])
 
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,50,50])
    upper2 = np.array([179,255,255])

    mask1 = cv2.inRange(hsv_image, lower1, upper1)
    mask2 = cv2.inRange(hsv_image, lower2, upper2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # Define lower and upper bounds for black color in HSV
    # For black colors in HSV  hue at maximum range (0 to 180), and saturation at maximum range (0 to 255) can play with the value 0 to 30 or 50 for black
    # if the image has very light black color then increase the value to 100 or less than 120
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 80, 80]) # black ko 70 - 90 raakhne # 70 maa alik madhuro kaalo chinxa tara background black noise pani tipxa kun kun maa

    # Create a mask for black color
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    multi_color_mask = cv2.bitwise_or(red_mask, black_mask)
    
    return multi_color_mask,red_mask,black_mask
    
def segment_and_classify(image_to_segment):
    license_characters = []
    # resize to our desired size
    image = cv2.resize(image_to_segment, (600,200))
    
    # saving the copy for later use
    original_image = image.copy()
    
    # converting the rgb image to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    ##?_________________Black and Red Masking_____________
    
    multi_color_mask, red_mask, black_mask = multi_color_masking(hsv_image)
    
    masked_image = cv2.bitwise_and(original_image, original_image, mask=multi_color_mask)
    
    ##?_________________Transform and crop_______________
    red_pixels = cv2.countNonZero(red_mask)
    black_pixels = cv2.countNonZero(black_mask)
    
    # calculating the dominant color
    if red_pixels > black_pixels:
        dominant_color = 'red'
        dominant_mask = red_mask
    else:
        dominant_color = 'black'
        dominant_mask = black_mask
        
    ##?_________________Contours of the dominant color_______________
    contours, hierarchy = cv2.findContours(dominant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area and get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate a minimum rectangle around the largest contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Convert box points to float32
        box = box.astype(np.float32)

        # Perform perspective transform
        transformed_image = perspective_transform(image, box)

        if transformed_image.shape[0] > transformed_image.shape[1]:
            transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # !Returning when no contour is found
        return license_characters
    
    transformed_resized_image = cv2.resize(transformed_image, (600, 200))
    final_original_image = cv2.cvtColor(transformed_resized_image, cv2.COLOR_BGR2RGB)
    
    ####?______________Processing to segment____________________________________
    # Grayscale
    image_process = cv2.cvtColor(final_original_image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Bluring
    kernel = np.ones((5,5),dtype=np.uint8)
    blurred_image = cv2.GaussianBlur(image_process, (5,5), 0)
    
    # Thresholding
    # thresh option 1
    # ret, thresh = cv2.threshold(image_process,130,255,cv2.THRESH_BINARY)

    # thresh option 2
    ret, thresh = cv2.threshold(blurred_image,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    ##____________________Addingg border_________________________
    border_size = 10
    border_color = (0, 0, 0)  # black color

    # Add border to the image
    cv2.rectangle(thresh, (0, 0), (thresh.shape[1], thresh.shape[0]), border_color, border_size)
    
    ##___________connected component analysis____________________
    _, labels = cv2.connectedComponents(thresh)
    # creating a black mask
    mask = np.zeros(thresh.shape, dtype="uint8")
    
    total_pixels = image.shape[0] * image.shape[1]
    lower = 300
    upper = total_pixels * 0.2
    
    ###____________Clean the image______________________________
    mask = filter_image(labels,thresh,mask,lower,upper)
    #TODO:
    # cv2.imwrite("cleanimage.jpg", mask)
    
    ##____________contours of characters_________________________
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 800 and cv2.contourArea(c) < 15000]

    # filtering with aspect ratio
    boundingBoxes = [bbox for bbox in boundingBoxes if bbox[2] / bbox[3] <= 3.5 and bbox[3] / bbox[2] <= 4]
    
    ## *sorting the bounding boxes
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 40:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
    
    
    ##_____________Segmenting the characters and detecting_____________________
    for i, bbox in enumerate(boundingBoxes):
        x, y, w, h = bbox
        if x > 5 and y > 5:
                padding = 5
                cropped_image = final_original_image[y - padding:y + h + padding, x - padding:x + w + padding]
                # ?our model is trained on 64x64 images
                cropped_resized = cv2.resize(cropped_image, (64,64))
                cropped_resized = cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR)
                
                # character = classify_character(cropped_resized)
                # license_characters.append(character)

                ## ?If we want binary image to classigy use this to  preprocess the cropped image
                gray_image = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
                # gray_image = cv2.resize(gray_image, (64,64))
                ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

                # Filter out small blobs
                min_blob_area = 400  # Minimum area threshold for blobs
                filtered_labels = labels.copy()
                for label in range(1, num_labels):
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area < min_blob_area:
                                filtered_labels[labels == label] = 0

                # Create the filtered image
                filtered_image = np.where(filtered_labels > 0, 255, 0).astype(np.uint8)

                # ## ?For thinning the characters
                kernel = np.ones((3,3), np.uint8)
                erode = cv2.erode(filtered_image, kernel, iterations=1)

                # get the recognized character from the classification model
                #TODO:
                # cv2.imwrite(f'./characters/character_{i}.png', erode)
                
                ## ? if image is eroded/grayscale it will be in (1,64,64,1) shape so we need to convert it to (1,64,64,3) for our model
                # character = classification_model.classify_character(cropped_resized)  #* Option1
                character = classification_model.classify_character(erode) #* Option2
                license_characters.append(character)
        else:
                cropped_image = final_original_image[y :y + h , x :x + w ]
                cropped_resized = cv2.resize(cropped_image, (64,64))
                cropped_resized = cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR)
                
                ## ? if image is eroded/grayscale it will be in (1,64,64,1) shape so we need to convert it to (1,64,64,3) for our model
                # character = classify_character(cropped_resized)
                character = classification_model.classify_character(erode)
                license_characters.append(character)
                
                # get the recognized character from the classification model
                # cv2.imwrite(f'./character_{i}.png', cropped_resized)
    # drawing the bounding boxes on segmented characters
    for bbox in boundingBoxes:
        x, y, w, h = bbox
        final_original_image = cv2.cvtColor(final_original_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(final_original_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #TODO:
        # cv2.imwrite("segmented_image.jpg", final_original_image)

    return license_characters, final_original_image


##########?_____________Testing the function________________________________________________  

# image = cv2.imread("./images/test.jpg")
# license_characters, final_segmented_image = segment_and_classify(image)
# cv2.imwrite("./characters/segmented_image.jpg", final_segmented_image)
# print(license_characters)
