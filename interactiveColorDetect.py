import cv2
import numpy as np

# Mouse callback function
def showPixelValue(event, x, y, flags, param):
    global img, combinedResult, placeholder

    if event == cv2.EVENT_MOUSEMOVE and img is not None:
        # Get the value of pixel from the location of mouse in (x, y)
        bgr = img[y, x]

        # Convert the BGR pixel into other color formats
        ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

        # Create an empty placeholder for displaying the values
        placeholder = np.zeros((img.shape[0], 400, 3), dtype=np.uint8)

        # Fill the placeholder with the values of color spaces
        cv2.putText(placeholder, "BGR {}".format(bgr), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "HSV {}".format(hsv), (20, 140), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "YCrCb {}".format(ycb), (20, 210), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "LAB {}".format(lab), (20, 280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

        # Combine the two results to show side by side in a single image
        combinedResult = np.hstack([img, placeholder])

        cv2.imshow('PRESS P for Previous, N for Next Image', combinedResult)

if __name__ == '__main__':
    # Load the image
    img_path = 'Images/sample4.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load the image: {img_path}")
        exit()

    # Resize the image to desired dimensions
    desired_width = 400
    desired_height = 800
    img = cv2.resize(img, (desired_width, desired_height))

    # Create a named window
    cv2.namedWindow('PRESS P for Previous, N for Next Image')

    # Set up the mouse callback function
    cv2.setMouseCallback('PRESS P for Previous, N for Next Image', showPixelValue)

    # Display the resized image
    cv2.imshow('PRESS P for Previous, N for Next Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
