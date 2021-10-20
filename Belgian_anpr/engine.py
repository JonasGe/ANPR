import cv2
import numpy as np


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def detect_belg(src):
    img, alpha, beta = automatic_brightness_and_contrast(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite('temp/blurred.png', src)

    crops = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h * 6 > w > 2 * h and h > 0.1 * w and w * h > img.shape[0] * img.shape[1] * 0.0001:
            crop = th[y:y + h, x:x + w]

            # Compute sum of white pixels
            white_summation = crop.sum()
            if white_summation > w * h * 0.4 * 255:
                # Compute sum of red pixel
                crop = img[y:y + h, x:x + w]
                crop_img = crop.astype('uint8')
                hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                lower_red = np.array([160, 100, 100])
                upper_red = np.array([179, 255, 255])
                red_mask = cv2.inRange(hsv, lower_red, upper_red)
                red_summation = red_mask.sum()

                if red_summation > 510:
                    crop_img = img[y:y + h, x - round(w / 10):x + w]
                    crop_img = crop_img.astype('uint8')
                    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    low_bleu = np.array([100, 150, 0])
                    high_bleu = np.array([140, 255, 255])
                    bleu_mask = cv2.inRange(hsv, low_bleu, high_bleu)
                    bleu_summation = bleu_mask.sum()

                    if bleu_summation > 255:

                        crop = gray[y:y + h, x:x + w]
                        crop_img = crop.astype('uint8')
                        th2 = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                    11, 2)

                        contours2, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        j = 0
                        for c in contours2:
                            area2 = cv2.contourArea(c)
                            x2, y2, w2, h2 = cv2.boundingRect(c)
                            if w2 * h2 > h * w * 0.01 and h2 > w2 and area2 < h * w * 0.9:
                                j += 1

                        if 12 > j > 4:
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            pts = np.array(box)
                            warped = four_point_transform(src, pts)
                            crops.append(warped)

                            cv2.drawContours(src, [box], 0, (0, 255, 0), 2)

                            image = cv2.imread('temp/blurred.png')
                            blurred_image = cv2.GaussianBlur(image, (43, 43), 30)
                            mask = np.zeros(image.shape, dtype=np.uint8)
                            channel_count = image.shape[2]
                            ignore_mask_color = (255,) * channel_count
                            cv2.fillPoly(mask, np.int32([box]), ignore_mask_color)
                            mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                            final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image, mask_inverse)
                            cv2.imwrite("temp/blurred.png", final_image)
    return src, crops


def process(src):

    # Brigthness and contrast adjustment
    adjusted, a, b = automatic_brightness_and_contrast(src)
    # BGR to gray
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    # Binary thresh
    #ret, th = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th
