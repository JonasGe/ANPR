from engine import process, detect_belg
import cv2
import argparse
import os
import glob
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--i', '-image', help="Input image path", type=str)

args = parser.parse_args()
abs_path = os.path.dirname(sys.executable)

if args.i:
    start = time.time()
    try:
        os.mkdir('temp')
    except:
        files = glob.glob('tmp')
        for f in files:
            os.remove(f)

    input_image = cv2.imread(args.i)
    detection, crops = detect_belg(input_image)

    i = 1
    for crop in crops:

        crop = process(crop)

        cv2.imwrite('temp/crop' + str(i) + '.jpg', crop)
        i += 1
    cv2.imwrite('temp/detection.jpg', detection)
    finish = time.time()
    print('Time processing >>>>>>  ' + str(finish - start))
