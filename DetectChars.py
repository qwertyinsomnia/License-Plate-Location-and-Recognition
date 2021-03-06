import os
import cv2
import numpy as np
from numpy.linalg import norm
import math
import random

import Preprocess
import Train

MODEL_SVM = 1
MODEL_CNN = 2

MIN_PIXEL_WIDTH = 3
MIN_PIXEL_HEIGHT = 12

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 100
# 
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0
# 

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

class PossibleChar:
    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break
    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
    MatchingChars = []

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            MatchingChars.append(possibleMatchingChar)

    return MatchingChars


def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)

    return fltAngleInDeg

def DetectCharsInPlates(listOfPossiblePlates, model, modelType, showSteps):
    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates

    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.Preprocess(possiblePlate.imgPlate, showSteps)
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if (len(listOfListsOfMatchingCharsInPlate) == 0):
            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) >= 5 and len(listOfListsOfMatchingCharsInPlate[i]) <= 8:
                possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, listOfListsOfMatchingCharsInPlate[i], model, modelType, showSteps)
        # intLenOfLongestListOfChars = 0
        # intIndexOfLongestListOfChars = 0

        # for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
        #     if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
        #         intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
        #         intIndexOfLongestListOfChars = i

        #         # suppose that the longest list of matching chars within the plate is the actual list of chars
        # longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        # possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate, model)
    return listOfPossiblePlates


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    contours = []
    imgThreshCopy = imgThresh.copy()
    height, width = imgThresh.shape
    # cv2.imshow("imgThreshCopy", imgThreshCopy)
    # cv2.waitKey(0)
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours counts:" + str(len(contours)))
    
    # if len(contours) == 53:
    #     imgContours = np.zeros((height, width, 3), np.uint8)
    #     for contour in contours:
    #         cnt = contour
    #         imgContours = cv2.drawContours(imgContours, [cnt], 0, (255, 255, 255), 1)
    #     cv2.imshow("imgContours", imgContours)
    #     cv2.waitKey(0)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        possibleChar = PossibleChar(contour)
        # if checkIfPossibleChar(possibleChar):
        if h > height * 0.3:
            listOfPossibleChars.append(possibleChar)

    print("contours after check:" + str(len(listOfPossibleChars)))
    # if len(listOfPossibleChars) == 19 or len(listOfPossibleChars) == 10 or len(listOfPossibleChars) == 9:
    #     imgContours = np.zeros((height, width, 3), np.uint8)
    #     for i in range(len(listOfPossibleChars)):
    #         cnt = listOfPossibleChars[i].contour
    #         imgContours = cv2.drawContours(imgContours, [cnt], 0, (255, 255, 255), 1)
    #     cv2.imshow("imgContours2", imgContours)
    #     cv2.waitKey(0)
    return listOfPossibleChars


def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)

    return listOfMatchingCharsWithInnerCharRemoved

def recognizeCharsInPlate(imgThresh, listOfMatchingChars, model, modelType, showSteps):
    predict_result = []
    strChars = ""

    height, width = imgThresh.shape
    # imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)
    # cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)

    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        # cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)

        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        contours, npaHierarchy = cv2.findContours(imgROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("contours counts when recognazing:" + str(len(contours)))
        charHeight = currentChar.intBoundingRectHeight
        charWidth = currentChar.intBoundingRectWidth
        s = charHeight * charWidth
        imgContours = np.zeros((charHeight, charWidth, 3), np.uint8)
        imgContours[:, :, 0] = imgROI
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (h < charHeight * 0.3 or w < charWidth * 0.3) or area < (s * 0.02):
                imgContours = cv2.drawContours(imgContours, [contour], 0, (0, 0, 0), -1)
        
        if showSteps:
            cv2.imshow("imgContours", imgContours)

        imgROI = imgContours[:, :, 0]
        if showSteps:
            cv2.imshow("imgROI", imgROI)
        
        if modelType == MODEL_CNN:
            imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            imgROIResized = cv2.morphologyEx(imgROIResized, cv2.MORPH_CLOSE, kernel)
            # imgROIResized = cv2.copyMakeBorder(imgROIResized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
            if showSteps:
                cv2.imshow("imgROIResized", imgROIResized)
                cv2.waitKey(0)
            npaROIResized = np.float32(npaROIResized)
            retval, npaResults, neigh_resp, dists = model.findNearest(npaROIResized, k = 1)
            strCurrentChar = str(chr(int(npaResults[0][0])))
            predict_result.append(strCurrentChar)

        elif modelType == MODEL_SVM:
            imgROIResized = cv2.resize(imgROI, (int(20/charHeight*charWidth * 1.2), 20))
            border = max(int(10 - 10/charHeight*charWidth * 1.2), 0)
            imgROIResized = cv2.copyMakeBorder(imgROIResized, 0, 0, border, border, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # imgROIResized = cv2.morphologyEx(imgROIResized, cv2.MORPH_CLOSE, kernel)
            if showSteps:
                cv2.imshow("imgROIResized", imgROIResized)
                cv2.waitKey(0)
            part_card = Train.preprocess_hog([imgROIResized])
            resp = model.predict(part_card)
            charactor = chr(int(resp[0]))
            predict_result.append(charactor)
        
    strChars = "".join(predict_result)
    print(strChars)

    return strChars