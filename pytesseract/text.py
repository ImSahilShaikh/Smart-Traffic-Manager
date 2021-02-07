import pytesseract
import cv2

#Addresing tesseract..
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#reading image
image = cv2.imread("4.jpg")

#converting image to text
text = pytesseract.image_to_string(image, lang='eng')

#removing Special symbols from text
text = ''.join(e for e in text if e.isalnum())

#Printing number
print("Number is : ",text)
cv2.waitKey(0)

#displaying orginal image
cv2.imshow("Image",image)
cv2.waitKey(0)


