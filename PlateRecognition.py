import cv2
import numpy as np
import os

import Preprocess
import DetectPlates
import DetectChars
import Train

MODEL_SVM = 1
MODEL_CNN = 2

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def PlateRecognition(path, showSteps, model, modelType):
    input  = cv2.imread(path)
    if input is None:
        print("error: unable to read image")
        os.system("pause")
        return

    input = Preprocess.PreResize(input)
    listOfPossiblePlates = DetectPlates.DetectPlates(input)

    listOfPossiblePlates = DetectChars.DetectCharsInPlates(listOfPossiblePlates, model, modelType)


    if len(listOfPossiblePlates) == 0:
        print("no license plates were detected")
        return "  "
    else:
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        print("\npossible plates output = " + str(len(listOfPossiblePlates)) + "\n")

        licPlate = listOfPossiblePlates[0]
        # for licPlate in listOfPossiblePlates:

        if showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program

        drawRedRectangleAroundPlate(input, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out

        # writeLicensePlateCharsOnImage(input, licPlate)           # write license plate text on the image
        if showSteps == True:
            cv2.imshow("input", input)                # re-show scene image
            cv2.waitKey(0)

        # cv2.imwrite("input.png", input)

    return licPlate.strChars

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    
    cv2.line(imgOriginalScene, (int(p2fRectPoints[0][0]), int(p2fRectPoints[0][1])), (int(p2fRectPoints[1][0]), int(p2fRectPoints[1][1])), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (int(p2fRectPoints[1][0]), int(p2fRectPoints[1][1])), (int(p2fRectPoints[2][0]), int(p2fRectPoints[2][1])), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (int(p2fRectPoints[2][0]), int(p2fRectPoints[2][1])), (int(p2fRectPoints[3][0]), int(p2fRectPoints[3][1])), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (int(p2fRectPoints[3][0]), int(p2fRectPoints[3][1])), (int(p2fRectPoints[0][0]), int(p2fRectPoints[0][1])), SCALAR_RED, 2)

def TestCase(filepath, model, modelType):
    rightOrWrong = 0 # right:1   wrong:0
    correct = filepath[:-4]
    filepath = "tests/" + filepath
    # print(correct)
    showSteps = False
    result = PlateRecognition(filepath, showSteps, model, modelType)
    if result == None:
        result = " "
    if result == correct:
        output = "correct!!!  read from image = " + result
        rightOrWrong = 1
    else:
        output = "error!!!  read from image = " + result + ", should be " + correct
        rightOrWrong = 0
    return output, rightOrWrong


def RunTestCases(model, modelType):
    rightNum = 0
    outputList = []
    for filename in os.listdir("tests"):
        if "." in filename:
            # print(filename) 
            output, rightOrWrong = TestCase(filename, model, modelType)
            outputList.append(output)
            rightNum += rightOrWrong

    for item in outputList:
        print(item)
    totalNum = len(outputList)
    print("among " + str(totalNum) + " tests, " + str(rightNum) + " passed!")


if __name__ == "__main__":
    filepath= "tests/LKZ3468.JPG"
    # filepath= "tests/1.png"
    showSteps = False
    modelType = MODEL_CNN
    if modelType == MODEL_SVM:
        t = Train.Trainer()
        model = t.train_svm()
    elif modelType == MODEL_CNN:
        t = Train.CNNTrainer()
        t.loadKNNDataAndTrainKNN()
        model = t.kNearest
        
    # PlateRecognition(filepath, showSteps, model, modelType)

    RunTestCases(model, modelType)