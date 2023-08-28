#TODO: detect corners of image
#TODO: transforming the perspective
import cv2
import shutil
import numpy as np
import imutils
import img2pdf
from PIL import Image
import os


class Scanner:
    def __init__(self,image):
        self.image = cv2.imread(image)
        shutil.copyfile(image, "backup.png")
        self.final_img = cv2.imread("backup.png")
    def start(self):
        # self.detect_document()
        self.warp_image()
    #function to make the document more easy to detect in the image
    def prepare_image(self):
        #turn black and white
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        #blur image
        blur = cv2.GaussianBlur(gray,  (5, 5), 0)
        #edged the image
        edged = cv2.Canny(blur, 75, 50)
        # cv2.imshow("title", edged)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return edged
    
    def detect_document(self):
        #make the thresholds on the image more clear
        sharpen = self.prepare_image()
        #find all the edges
        contours = cv2.findContours(sharpen.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        grabbed = imutils.grab_contours(contours)
        sortedContours = sorted(grabbed, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        #go through the list of edges and returns the one that forms a square
        for contour in sortedContours:
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # If approx. contour has four points, then we can assume that we have found the document
            if len(approximation) == 4:
                screenCnt = approximation
                #draw the edges
                cv2.drawContours(self.image, [screenCnt], -1, (0, 255, 0), 2)
                # cv2.imshow("title", self.image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                break
        corners = []
        for d in screenCnt:
            tuple_point = tuple(d[0])
            cv2.circle(self.image, tuple_point, 3, (0, 0, 255), 4)
            corners.append(tuple_point)

        cv2.imshow('Circled corner points', self.image)
        cv2.waitKey(0)
        return screenCnt
    def order_points(self,pts):
        # initializing the list of coordinates to be ordered
        rect = np.zeros((4, 2), dtype = "float32")
        print(pts)
        s = pts.sum(axis = 1)

        # top-left point will have the smallest sum
        rect[0] = pts[np.argmin(s)]

        # bottom-right point will have the largest sum
        rect[2] = pts[np.argmax(s)]

        '''computing the difference between the points, the
        top-right point will have the smallest difference,
        whereas the bottom-left will have the largest difference'''
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # returns ordered coordinates
        print(rect)
        return pts
    
    def warp_image(self):
        corners = self.detect_document()
        # unpack the ordered coordinates individually
        # ratio = self.image.shape[0] / 500.0
        rect = self.order_points(corners.reshape(4, 2))
        (tl, bl, br, tr) = rect
        #tl bl br tr
        print(rect)
        print("top left", tl)
        print("bottom left", bl)
        print("bottom right", br)
        print("top right", tr)
        '''compute the width of the new image, which will be the
        maximum distance between bottom-right and bottom-left
        x-coordinates or the top-right and top-left x-coordinates'''
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        '''compute the height of the new image, which will be the
        maximum distance between the top-left and bottom-left y-coordinates'''
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        '''construct the set of destination points to obtain an overhead shot'''
        dst = np.array([
            [0, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1],
            [maxWidth - 1, 0]], dtype = "float32")
        corners = corners.astype(np.float32)
        dst = dst.astype(np.float32)

        # compute the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst)

        # Apply the transform matrix
        warped = cv2.warpPerspective(self.final_img, transform_matrix, (maxWidth, maxHeight))
        # warped_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # bigger = cv2.resize(warped, (self.image.shape[0],self.image.shape[1]), interpolation=cv2.INTER_CUBIC)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(warped, -1, sharpen_kernel)
        # blurred = cv2.GaussianBlur(warped, (0, 0), 3)

        # sharpen = cv2.addWeighted(warped, 1.5, blurred, -0.5, 0)

        cv2.imwrite('./'+'scan'+'.png',sharpen)
        self.convert_pdf("scan.png")
        cv2.imshow("Warped Image", sharpen)
        cv2.imshow("title", self.image)
        cv2.waitKey(0)
    def convert_pdf(self, image):
        # opening image
        image = Image.open(image)
        pdf_path = "scanned.pdf"
        
        # converting into chunks using img2pdf
        pdf_bytes = img2pdf.convert(image.filename)
        
        # opening or creating pdf file
        file = open(pdf_path, "wb")
        
        # writing pdf files with chunks
        file.write(pdf_bytes)
        
        # closing image file
        image.close()
        
        # closing pdf file
        file.close()
        
        # output
        print("Successfully made pdf file")
        
test = Scanner(image="example.jpg")
test.start()