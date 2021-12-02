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
    listOfPossiblePlates = DetectPlates.DetectPlates(input, showSteps)

    listOfPossiblePlates = DetectChars.DetectCharsInPlates(listOfPossiblePlates, model, modelType, showSteps)


    if len(listOfPossiblePlates) == 0:
        print("no license plates were detected")
        return "  "
    else:
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        print("\npossible plates output = " + str(len(listOfPossiblePlates)) + "\n")

        licPlate = listOfPossiblePlates[0]
        # for licPlate in listOfPossiblePlates:

        if showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n\n")
            return "", licPlate.imgPlate

        drawRedRectangleAroundPlate(input, licPlate)

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")

        if showSteps == True:
            writeLicensePlateCharsOnImage(input, licPlate)

        if showSteps == True:
            cv2.imshow("input", input)
            cv2.waitKey(0)

    return licPlate.strChars, licPlate.imgPlate

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 20.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_RED, intFontThickness)

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
    result, roi = PlateRecognition(filepath, showSteps, model, modelType)
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
    filepath= "tests/PLD2460.JPG"
    # filepath= "tests/MCLRNF1.png"

    # testCaseFlag = True
    testCaseFlag = False

    modelType = MODEL_SVM
    if modelType == MODEL_SVM:
        t = Train.Trainer()
        model = t.train_svm()
    elif modelType == MODEL_CNN:
        t = Train.CNNTrainer()
        t.loadKNNDataAndTrainKNN()
        model = t.kNearest

    if not testCaseFlag:
        showSteps = True
        PlateRecognition(filepath, showSteps, model, modelType)
    else:
        RunTestCases(model, modelType)
