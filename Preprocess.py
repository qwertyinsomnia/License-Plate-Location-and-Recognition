import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def PreResize(input):
    height, width, numChannels = input.shape
    if height > 1080 or width > 1920:
        rszRatio = max(height / 1080, width / 1920)
        dstHeight = int(height / rszRatio)
        dstWidth = int(width / rszRatio)
        output = cv2.resize(input, (dstWidth, dstHeight), interpolation=cv2.INTER_LANCZOS4)
        return output
    else:
        return input


def Preprocess(input, showSteps):
    height, width, numChannels = input.shape
    # if height > 1080 or width > 1920:
    #     rszRatio = max(height / 1080, width / 1920)
    #     dstHeight = int(height / rszRatio)
    #     dstWidth = int(width / rszRatio)
    #     input = cv2.resize(input, (dstWidth, dstHeight), interpolation=cv2.INTER_LANCZOS4)
    imgGrayscale = GetHSV(input, height, width)
    # imgGrayscale = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # imgMaxContrastGrayscale = MaximizeContrast(imgGrayscale, height, width)
    imgMaxContrastGrayscale = cv2.equalizeHist(imgGrayscale)

    # cv2.imshow("test", imgMaxContrastGrayscale)
    # cv2.waitKey(0)
    # imgBlurred = cv2.medianBlur(imgMaxContrastGrayscale, 3)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    if showSteps:
        cv2.imshow("imgGrayscale.png", imgGrayscale)
        cv2.imshow("imgMaxContrastGrayscale.png", imgMaxContrastGrayscale)
        cv2.imshow("plateimgBlurred.png", imgBlurred)
        cv2.imshow("plateimgThresh.png", imgThresh)
        cv2.waitKey(0)
    return imgGrayscale, imgThresh

def GetHSV(input, height, width):
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue
	
def MaximizeContrast(imgGrayscale, height, width):
    # imgTopHat = np.zeros((height, width, 1), np.uint8)
    # imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat