import cv2
import numpy as np
import math
import random

# import PlateRecognition
import Preprocess
import DetectChars

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

class PossiblePlate:
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""

def DetectPlates(input):
    listOfPossiblePlates = []
    height, width, numChannels = input.shape

    # imgGrayscale = np.zeros((height, width, 1), np.uint8)
    # imgThresh = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    imgGrayscale, imgThresh = Preprocess.Preprocess(input)

    listOfPossibleChars = findPossibleChars(imgThresh)

    print("\n" + str(len(listOfPossibleChars)) + " possible chars found")

    listOfMatchingChars = DetectChars.findListOfListsOfMatchingChars(listOfPossibleChars)
    # print(listOfMatchingChars)

    for listOfMatchingChars in listOfMatchingChars:
        possiblePlate = extractPlate(input, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:
            p2fRectPoints = cv2.boxPoints(possiblePlate.rrLocationOfPlateInScene)
            w = (abs(p2fRectPoints[3][1] - p2fRectPoints[0][1]) + abs(p2fRectPoints[2][1] - p2fRectPoints[1][1])) / 2
            h = (abs(p2fRectPoints[0][0] - p2fRectPoints[1][0]) + abs(p2fRectPoints[3][0] - p2fRectPoints[2][0])) / 2
            if h == 0:
                continue
            aspect_ratio = float(w) / h
            if aspect_ratio > 1.6 and aspect_ratio < 4.5:
                listOfPossiblePlates.append(possiblePlate)

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")
    return listOfPossiblePlates


def findPossibleChars(imgThresh):
    listOfPossibleChars = []
    imgThreshCopy = imgThresh.copy()

    contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for item in contours:
        possibleChar = DetectChars.PossibleChar(item)
        if DetectChars.checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate()

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + 
                    listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - 
                    listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = (tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped

    return possiblePlate