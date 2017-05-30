import cv2
import numpy as np
import operator
import os


CONTOUR_AREA = 100

IMAGE_WIDTH = 20
IMAGE_HEIGHT = 30

class ContourData():

   
    nContour = None           
    boundRect = None        
    rectX = 0                
    rectY = 0                
    rectW = 0            
    rectH = 0           
    FloatArea = 0.0              
                                                      
    def calc(self):              
        [intX, intY, intW, intH] = self.boundRect
        self.rectX = intX
        self.rectY = intY
        self.rectW = intW
        self.rectH = intH

    def check(self):                            
        if self.FloatArea < CONTOUR_AREA: return False        
        return True


def main():
    contourData1= []                
    validdata1 = []             

    try:
        nclassifications = np.loadtxt("classifications.txt", np.float32)                  
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    
                                                                                     
    try:
        nflattenedimg = np.loadtxt("flattened_images.txt", np.float32)                 
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    

    nclassifications = nclassifications.reshape((nclassifications.size, 1))       

    knear = cv2.ml.Knear_create()                   

    kNearest.train(nflattenedimg, cv2.ml.ROW_SAMPLE, nclassifications)

    imgtestno = cv2.imread("test1.png")          

    if imgtestno is None:                           
        print ("error: image not read from file \n\n"   )     
        os.system("pause")                                  
        return                                              
  
    imGray = cv2.cvtColor(imgtestno, cv2.COLOR_BGR2GRAY)       
    imblur = cv2.GaussianBlur(imGray, (5,5), 0)  
                                                                                     
                                                        
    imgThresh = cv2.adaptiveThreshold(imblur,                           
                                      255,                                  
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                      cv2.THRESH_BINARY_INV,                
                                      11,                                   
                                      2)                                    

    imThreshC = imgThresh.copy()        

    imgContours, nContours, npaHierarchy = cv2.findContours(imThreshC,             
                                                 cv2.RETR_EXTERNAL,         
                                                 cv2.CHAIN_APPROX_SIMPLE)   

    for nContour in nContours:                             
        ContourData = ContourData()                                             
        ContourData.nContour = nContour                                         
        ContourData.boundRect = cv2.boundRect(ContourData.nContour)     
        ContourData.calc()                    
        ContourData.FloatArea = cv2.contourArea(ContourData.nContour)           
        allContoursWithData.append(ContourData)                                     
    
    for ContourData in allContoursWithData:                 
                                                                                      
    if ContourData.check():             
            validdata1.append(ContourData)       
        
    

    validdata1.sort(key = operator.attrgetter("rectX"))         

    strFinalString = ""         

    for ContourData in validdata1:            
                                                
        cv2.rectangle(imgtestno,                                        
                      (ContourData.rectX, ContourData.rectY),     
                      (ContourData.rectX + ContourData.rectW, ContourData.rectY + ContourData.rectH),      
                      (255, 0,0),              
                      2)                        

        iROI = imgThresh[ContourData.rectY : ContourData.rectY + ContourData.rectH,     
                           ContourData.rectX : ContourData.rectX + ContourData.rectW]

        iROIResized = cv2.resize(iROI, (IMAGE_WIDTH, IMAGE_HEIGHT))             
                                                                       
        nROIResize = iROIResized.reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT))      

        nROIResize = np.float32(nROIResize)       

        retval, npaResults, neigh_resp, dists = knear.findNearest(nROIResize, k = 1)     

        strCurrentChar = str(chr(int(npaResults[0][0])))                                            

        strFinalString = strFinalString + strCurrentChar            
   

    print ("\n" + strFinalString + "\n")                  

    cv2.imshow("imgtestno", imgtestno)      
    cv2.waitKey(0)                                          

    cv2.destroyAllWindows()             

    return

if __name__ == "__main__":
    main()
