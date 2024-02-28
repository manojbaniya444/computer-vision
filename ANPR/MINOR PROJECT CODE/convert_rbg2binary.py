import cv2
import os
import numpy as np

def convert_to_binary_image(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path)

            # Convert to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Resize the image
            gray_image = cv2.resize(gray_image, (128, 128))

            # Apply binary thresholding
            ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

            # Filter out small blobs
            min_blob_area = 400
            filtered_labels = labels.copy()
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < min_blob_area:
                    filtered_labels[labels == label] = 0

            # Create the filtered image
            filtered_image = np.where(filtered_labels > 0, 255, 0).astype(np.uint8)

            # Apply erosion
            kernel = np.ones((7, 7), np.uint8)
            erode = cv2.erode(filtered_image, kernel, iterations=1)

            # Save the result
            output_path = os.path.join(output_folder, f'binary_{filename}')
            cv2.imwrite(output_path, erode)

if __name__ == "__main__":
    input_folder = '/path/to/input/folder'
    output_folder = '/path/to/output/folder'
    convert_to_binary_image(input_folder, output_folder)
