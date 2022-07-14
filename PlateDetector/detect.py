import os
import cv2
import imutils

# Define paths
input_path = "imgs/"
output_path = "outp/"

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
        self.xmin:int = None
        self.xmax:int = None
        self.ymin:int = None
        self.ymax:int = None

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
        bfilter = cv2.bilateralFilter(self.img_gray, 11, 17, 17)
        bfilter = cv2.GaussianBlur(bfilter, (3, 3), 0)
        #Edge detection
        self.edged = cv2.Canny(bfilter, 30, 200)

        # cv2.imshow("Edged", self.edged)
        # cv2.waitKey(0)
        return

    # Function to find contours
    def find_contours(self) -> None:
        # Find contours
        cnts = cv2.findContours(self.edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        im1 = self.img.copy()
        # im1 = cv2.drawContours(im1, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("Contours", im1)
        # cv2.waitKey(0)

        # for cnt in cnts:
        #     im1 = cv2.drawContours(im1, [cnt], -1, (0, 255, 0), 3)
        #     cv2.imshow("Contours", im1)
        #     cv2.waitKey(0)

        # print(cnts[0])
        # approx = cv2.approxPolyDP(cnts[0], 0.01 * cv2.arcLength(cnts[0], True), True)
        # print('approx:', approx)
        # im1 = cv2.drawContours(im1, [approx], -1, (0, 255, 0), 3)
        # cv2.imshow("Contours", im1)
        # cv2.waitKey(0)

        # Take the biggest contour as the license plate
        # self.cropped = self.img_gray[approx[:, 0, 1].min():approx[:, 0, 1].max(), approx[:, 0, 0].min():approx[:, 0, 0].max()]
        # cv2.imshow("Cropped", self.cropped)
        # cv2.waitKey(0)

        # # Get the best contour found (with 4 corners)
        for cnt in cnts:
            # Calculate perimeter of the contour and approximate it with a polygon
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.001 * perimeter, True)

            im1 = self.img.copy()
            # im1 = cv2.drawContours(im1, [approx], -1, (0, 255, 0), 3)
            # cv2.imshow("Contours", im1)

            # Take the biggest contour found
            big = self.img_gray[approx[:, 0, 1].min():approx[:, 0, 1].max(), approx[:, 0, 0].min():approx[:, 0, 0].max()]
            big_ratio = big.shape[1] / big.shape[0]
            big_area = big.shape[0] * big.shape[1]
            # print('big_ratio:', big_ratio)
            # print('big_area:', big_area)
            # cv2.waitKey(0)

            if big_ratio >= 3.5 and big_ratio <= 5.5 and big_area > 2000:
                im1 = cv2.drawContours(im1, [approx], -1, (0, 255, 0), 3)
                # cv2.imshow("Contours", im1)
                # cv2.waitKey(0)

                # Take the biggest contour as the license plate
                self.xmin = approx[:, 0, 0].min()
                self.xmax = approx[:, 0, 0].max()
                self.ymin = approx[:, 0, 1].min()
                self.ymax = approx[:, 0, 1].max()
                self.cropped = self.img_gray[self.ymin:self.ymax, self.xmin:self.xmax]
                # self.cropped = self.img_gray[approx[:, 0, 1].min():approx[:, 0, 1].max(), approx[:, 0, 0].min():approx[:, 0, 0].max()]
                # cv2.imshow("Cropped", self.cropped)
                # cv2.waitKey(0)

                break

        self.result = self.img.copy()
        self.result = cv2.rectangle(self.result, (self.xmin, self.ymin), (self.xmax, self.ymax), (0, 255, 0), 3)

        # cv2.imshow("Cropped", self.cropped)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Function to write the OCR string to the image
    def write_ocr(self, ocr_string:str) -> None:
        write_point = (self.xmin, self.ymin) - (0, 10)

        self.result = cv2.putText(self.img,
            text = ocr_string,
            org = write_point,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0, 255, 0),
            thickness = 2,
            lineType = cv2.LINE_AA)

        self.result = cv2.rectangle(self.img,
            pt1 = (self.xmin, self.ymin),
            pt2 = (self.xmax, self.ymax),
            color = (0, 255, 0),
            thickness = 3)
        
        cv2.imshow("Result", self.result)
        cv2.waitKey(0)
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

d = PlateDetect()
# Scan all images in the input directory
for filename in os.listdir(input_path):
    # if filename.endswith(".jpg"):
    print(filename)
    image_path = input_path + filename
    result_path = output_path + filename
    d.detect(image_path)
    d.save_image(result_path)

# d.detect('images/test.jpg')