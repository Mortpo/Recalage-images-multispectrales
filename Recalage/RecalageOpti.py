import cv2
import numpy as np
import os
import SimpleITK as sitk
import sys
import glob
from joblib import Parallel, delayed
from tqdm import tqdm,trange

def undistordImage(image:str,Cameras:tuple,paramCam:dict):

    for cam in Cameras:
        if cam in image:
            mtx = paramCam[cam]["mtx"]
            dist = paramCam[cam]["dist"]
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,  w = img.shape[:2]
            
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))
            # undistort
            undistorded = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            roiundistorded = undistorded[y:y+h, x:x+w]
            roiundistorded = cv2.resize(roiundistorded,(2567, 1893),cv2.INTER_CUBIC)
            #cv2.imwrite("undistord"+cam+".bmp",undistorded)
            return roiundistorded


def RecalageOpenCV(imageRef: any ,imageToAligned: any):
    # Convert to grayscale.
    height, width = imageRef.shape
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    
    kp1, d1 = orb_detector.detectAndCompute(imageToAligned, None)
    kp2, d2 = orb_detector.detectAndCompute(imageRef, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    
    # Sort matches on the basis of their Hamming distance.
    matches = sorted (matches,key = lambda x: x.distance)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(imageToAligned,
                        homography, (width, height))

    #computeMetric(sitk.GetImageFromArray(np.float32(imageToAligned))  ,sitk.GetImageFromArray(np.float32(transformed_img)))

    # Save the output.
    #cv2.imwrite(saveName, transformed_img)
    return transformed_img

DataSetPath = os.path.join("/home","mortpo","ESIREM","PFE","DataSet")
RecalagePath = os.path.join("/home","mortpo","ESIREM","PFE","Recalage")
SavePath =  os.path.join(".")

Cameras = ("Camera1","Camera2","Camera3","Camera4")
CHEMINSET = ("SET1","SET2","SET3","SET4")
#CHEMINSETP = ("SET1","SET2","SET3","SET4","SET5","SET6","SET7","SET8","SET9","SET10","SET11","SET12","SET13")
BASESET={"DAMAV_Jeux_Image":CHEMINSET}#,"Parallax_Images":CHEMINSETP}


cheminParam =[]
regex = os.path.join(RecalagePath,"Calibration","Data","**","*.npz")
cheminParam = sorted(glob.glob(regex,recursive=True))
paramCam = dict()
for cam in Cameras:
    for fichierParam in cheminParam:
        if cam in fichierParam:
            with np.load(fichierParam) as data:
                paramCam[cam] = {"mtx" : data['mtx'],"dist" : data['dist']}

for key in BASESET:
    setTuple = BASESET[key]
    for set in tqdm(setTuple):
        CHEMINSET = os.path.join(key,set)
        cheminimage = os.path.join(DataSetPath,CHEMINSET,"*.bmp")
        CHEMINIMAGES = sorted(glob.glob(cheminimage))

        undistordedImg = Parallel(n_jobs=4,prefer="threads")(delayed(undistordImage)(CHEMINIMAGES[i],Cameras,paramCam)for i in range(len(CHEMINIMAGES)))
        homography = Parallel(n_jobs=4,prefer="threads")(delayed(RecalageOpenCV)(undistordedImg[0],undistordedImg[i])for i in range(1,len(CHEMINIMAGES)))
        
        images = []
        for imageRecale in homography:
            images.append(imageRecale)
        images.append(undistordedImg[0])

        zeros = np.zeros((images[0].shape),dtype= "uint8" )

        mask = []
        for imageRecale in images:
            mask.append(cv2.compare(imageRecale,zeros,cv2.CMP_EQ))

        fusion = zeros
        for i in range(0,len(mask)):
            fusion = cv2.bitwise_or(mask[i],fusion)
        masks = cv2.bitwise_not(fusion)

        imagemasked = []
        for imgeRecale in images:
            imagemasked.append(cv2.bitwise_and(imgeRecale,masks))
            

        #cv2.imwrite("mask.jpg",masks)
        end = cv2.merge(imagemasked)
        cv2.imwrite(os.path.join(SavePath,key+set+"outcolor.jpg"),end)
