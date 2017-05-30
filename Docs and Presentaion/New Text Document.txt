import sys
import cv2
import os
import numpy as np
Contour_Area = 100
image_height = 30
image_width = 30
                                                                           
def main():
    img = cv2.imread("training.png")

    if img is None:
        print("error:no image\n\n")
        os.system("pause")
        return

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur =cv2.GaussianBlur(gray,(5,5),0)
    thresh =cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,11,2)

    cv2.imshow("thresh",thresh)

    threshcopy = thresh.copy()

    contours,ncontours,nhierarchy = cv2.findContours(threshcopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    flattenedimages = np.empty((0,image_width*image_height))

    intclassifications =[]
                                                               
    intvalidchar = [ ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'),ord('9'),ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'), ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T') ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
    for ncontour in ncontours:
        if cv2.contourArea(ncontour) > Contour_Area:
            [intX,intY,intW,intH] = cv2.boundRect(ncontour)

            cv2.rectangle(img,(intX,intY),(intX+intW,intY+intH),(0,0,255),2)


            iROI = thresh [intY:intY+intH, intX:intX+intW]
            iROIResized = cv2.resize (iROI, (image_width, image_height))


            cv2.imshow("iROI",iROI)
            cv2.imshow("iROIResized",iROIResized)
            cv2.imshow("trainging_numbers.png",img)

            intChar = cv2.waitKey(0)

            if intChar == 27:
                sys.exit()
            elif intChar in intvalidchar:
                                                                                
                intclassifications.append(intChar)

                flattenedimage = iROIResized.reshape(1,image_width*image_height)

                flattenedimages = np.append(flattenedimages,flattenedimage,0)


    fclassification = np.array(intClassifications,np.float32)
    nclassification = fclassification.reshape(fclassification.size,1)

    print("\n\ntraining done!!\n")

    np.savetext ("claasification.txt",fclassification)
    np.savetext ("flattened_images.txt",flattenedimages)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main() 
                                         End of Program ##############################################################################
