#import packages
import cv2
import imutils as im

# Read the image file
input = 'images/t8.jpg'
image = cv2.imread(input)

# Resize the image - change width to 500
newwidth = 500
image = im.resize(image, width=newwidth)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
d, sigmaColor, sigmaSpace = 11,17,17
filtered_img = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

# Find Edges of the grayscale image
lower, upper = 170, 200
edged = cv2.Canny(filtered_img, lower, upper)

# Find contours based on Edges
cnts,hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
NumberPlateCnt = None
print("Number of Contours found : " + str(len(cnts)))


# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        
        epsilon = 0.01 * peri
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) == 4:  # Select the contour with 4 corners
            print(approx)
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break

# Display the original image
cv2.imshow("Input Image", image)
# Display Grayscale image
cv2.imshow("Gray scale Image", gray)
# Display Filtered image 
cv2.imshow("After Applying Bilateral Filter", filtered_img)
# Display Canny Image
cv2.imshow("After Canny Edges", edged)
# cv2.waitKey(0)
# Drawing the selected contour on the original image
cv2.drawContours(image, cnts, -1, (255,0,0), 2)
if NumberPlateCnt is not None:
    print("Number Plate Found")
    cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 2)
cv2.imshow("Output", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed