import os
import cv2
import imutils

# Define paths
input_path = "images/"
output_path = "output/"

# Class PlateDetect
class PlateDetect:
    # Constructor
    def __init__(self) -> None:
        # Input images
        self.img:cv2.Mat = None
        self.img_gray:cv2.Mat = None

        # Detection results
        self.edged:cv2.Mat = None
        self.cropped:cv2.Mat = None
        self.location:list[list[int]] = None

        # Output image with OCR string written on it
        self.result:cv2.Mat = None
        return

    # Function to read an image
    def read_image(self, image_path:str) -> None:
        self.img = cv2.imread(image_path)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Gray", self.img_gray)
        # cv2.waitKey(0)
        return

    # Function to edge detection
    def edge_detection(self) -> None:
        #Noise reduction
        bfilter = cv2.bilateralFilter(self.img, 11, 17, 17)
        #Edge detection
        self.edged = cv2.Canny(bfilter, 30, 200)

        # cv2.imshow("Edged", self.edged)
        # cv2.waitKey(0)
        return

    # Function to find contours
    def find_contours(self) -> None:
        # Find contours
        cnts = cv2.findContours(self.edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        # Get the best contour found (with 4 corners)
        for cnt in cnts:
            approx = cv2.approxPolyDP(cnt, 10, True)
            if len(approx) == 4:
                self.location = approx
                break

        # Get the contour coordinates
        y1, x1 = self.location[0][0]
        y2, x2 = self.location[2][0]
        self.cropped = self.img_gray[x1:x2+1, y1:y2+1]

        # cv2.imshow("Cropped", self.cropped)
        # cv2.waitKey(0)
        return

    # Function to write the OCR string to the image
    def write_ocr(self, ocr_string:str) -> None:
        write_point = self.location[0][0] - (0, 10)

        self.result = cv2.putText(self.img,
            text=ocr_string,
            org=write_point,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA)

        self.result = cv2.rectangle(self.img,
            pt1=self.location[0][0],
            pt2=self.location[2][0],
            color=(0, 255, 0),
            thickness=3)
        
        # cv2.imshow("Result", self.result)
        # cv2.waitKey(0)
        return

    # Function to save the image
    def save_image(self, image_path:str) -> None:
        cv2.imwrite(image_path, self.result)
        return

    # Function to execute all functions in order
    def detect(self, image_path:str) -> cv2.Mat:
        self.read_image(image_path)
        self.edge_detection()
        self.find_contours()
        return self.cropped


# Driver function
def driver() -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return
