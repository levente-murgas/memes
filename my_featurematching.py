import cv2
import numpy as np
import os
from multiprocessing import Pool
import time
import imutils
from datetime import datetime
from tqdm.contrib.concurrent import  thread_map  # or thread_map
from line_profiler import LineProfiler
from tqdm import tqdm

def processimgs(template_target):

    basepath = 'D:/'
    meme_dir = "D:/Memes2023_splitted_clean"

    templates = os.listdir(meme_dir)

    orb = cv2.ORB_create()

    fileloc1 = os.path.join(meme_dir, template_target)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1, None)

    if type(des1) is np.ndarray:

        matchdir = 'matches'

        donepath = os.path.join('D:', matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.txt' in file:
                    doneitem = file.replace('.txt','.jpg')
                    done.append(doneitem)

        processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
        processfile = open(processfileloc,"w")
        processfile.close()

        matched_images = []
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
        flann = cv2.FlannBasedMatcher(index_params)

        for template_compare in templates:
            if template_target != template_compare:
                if template_compare not in done:
                    try:
                        fileloc2 = os.path.join(meme_dir, template_compare)
                        img2 = cv2.imread(fileloc2)
                        img2 = imutils.resize(img2, width=800)
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                        kp2, des2 = orb.detectAndCompute(img2, None)

                        if type(des2) is np.ndarray:
                            matches = flann.match(des1, des2)
                            goodmatches = []
                            for match in matches:
                                if match.distance <= 25:
                                    goodmatches.append(match)

                            if len(goodmatches) >= 10:
                                matched_images.append(template_compare)
                    except:
                        continue
        
        if len(matched_images) > 0:
            with open (processfileloc, 'a') as f:
                for item in matched_images:
                    f.write("%s\n" % item)

if __name__ == '__main__':

    start = time.time()

    meme_dir = "D:/Memes2023_splitted_clean"

    templatesbase = os.listdir(meme_dir)

    r = thread_map(processimgs, templatesbase)

    runtime = time.time()-start
    print('Script runtime:',round(runtime,2),'seconds',round(runtime/3600,2),'hours')
