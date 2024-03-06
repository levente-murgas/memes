import cv2
import numpy as np
import os
from multiprocessing import Pool
import time
import imutils
from datetime import datetime
from tqdm.contrib.concurrent import thread_map, process_map  # or thread_map
from line_profiler import LineProfiler
import cProfile
import timeit
import shutil
def processimgs_cpu(template_target):

    print('Processing this:',template_target)

    basepath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(basepath, "source_images_clean")

    templates = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templates.append(file)

    orb = cv2.ORB_create()

    fileloc1 = os.path.join(filepath, template_target)
    # print(fileloc1)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)


    if type(des1) is np.ndarray:

        matchdir = 'matches'

        donepath = os.path.join(basepath, matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.DS_Store' not in file:
                    if '.txt' in file:
                        doneitem = file.replace('.txt','.jpg')
                        done.append(doneitem)

        # print(done)

        print('Processing',template_target,'-',len(done),'in process/done','-',datetime.now())

        for indexb,template_compare in enumerate(templates):
            if template_target != template_compare:
                if template_compare not in done:

                    processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
                    processfile = open(processfileloc,"w")
                    processfile.close()

                    fileloc2 = os.path.join(filepath, template_compare)
                    #print(fileloc2)
                    img2 = cv2.imread(fileloc2)
                    img2 = imutils.resize(img2, width=800)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(img2, None)

                    if type(des2) is np.ndarray:

                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)

                        goodmatches = []

                        for match in matches:
                            if match.distance <= 25:
                                goodmatches.append(match)

                        if len(goodmatches) >= 10:
                            print('Matched:',template_target,template_compare)
                            matching_result = cv2.drawMatches(img1, kp1, img2, kp2, goodmatches, None, flags=2)
                            targetdir = matchdir
                            savename = template_target.replace('.jpg','') + '_' + template_compare.replace('.jpg','') + '_' + str(len(goodmatches)) + '.jpg'
                            fileloc3 = os.path.join(basepath, targetdir, savename)
                            cv2.imwrite(fileloc3, matching_result)

def processimgs_cpu_flann(template_target):

    # print('Processing this:',template_target)

    basepath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(basepath, "source_images_clean")

    templates = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templates.append(file)

    orb = cv2.ORB_create()

    fileloc1 = os.path.join(filepath, template_target)
    # print(fileloc1)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)


    if type(des1) is np.ndarray:

        matchdir = 'matches'

        donepath = os.path.join(basepath, matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.DS_Store' not in file:
                    if '.txt' in file:
                        doneitem = file.replace('.txt','.jpg')
                        done.append(doneitem)

        # print(done)

        # print('Processing',template_target,'-',len(done),'in process/done','-',datetime.now())

        for indexb,template_compare in enumerate(templates):
            if template_target != template_compare:
                if template_compare not in done:

                    processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
                    processfile = open(processfileloc,"w")
                    processfile.close()

                    fileloc2 = os.path.join(filepath, template_compare)
                    #print(fileloc2)
                    img2 = cv2.imread(fileloc2)
                    img2 = imutils.resize(img2, width=800)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(img2, None)

                    if type(des2) is np.ndarray:

                        FLANN_INDEX_LSH = 6
                        index_params= dict(algorithm = FLANN_INDEX_LSH,
                                        table_number = 6, # 12
                                        key_size = 12,     # 20
                                        multi_probe_level = 1) #2
                        flann = cv2.FlannBasedMatcher(index_params)
                        matches = flann.match(des1, des2)

                        goodmatches = []

                        for match in matches:
                            if match.distance <= 25:
                                goodmatches.append(match)

                        if len(goodmatches) >= 10:
                            # print('Matched:',template_target,template_compare)
                            matching_result = cv2.drawMatches(img1, kp1, img2, kp2, goodmatches, None, flags=2)
                            targetdir = matchdir
                            savename = template_target.replace('.jpg','') + '_' + template_compare.replace('.jpg','') + '_' + str(len(goodmatches)) + '.jpg'
                            fileloc3 = os.path.join(basepath, targetdir, savename)
                            cv2.imwrite(fileloc3, matching_result)

def processimgs_cpu_flann_optimized(template_target):

    basepath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(basepath, "source_images_clean")

    templates = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templates.append(file)

    orb = cv2.ORB_create()

    fileloc1 = os.path.join(filepath, template_target)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)


    if type(des1) is np.ndarray:

        matchdir = 'matches'

        donepath = os.path.join(basepath, matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.DS_Store' not in file:
                    if '.txt' in file:
                        doneitem = file.replace('.txt','.jpg')
                        done.append(doneitem)


        processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
        processfile = open(processfileloc,"a")
        processfile.close()

        matched_images = []
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
        flann = cv2.FlannBasedMatcher(index_params)

        for indexb,template_compare in enumerate(templates):
            if template_target != template_compare:
                if template_compare not in done:

                    fileloc2 = os.path.join(filepath, template_compare)
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

        if len(matched_images) > 0:
            with open (processfileloc, 'a') as f:
                for item in matched_images:
                    f.write("%s\n" % item)

def processimgs_gpu(template_target):

    basepath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(basepath, "source_images_clean")

    templates = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templates.append(file)

    orb = cv2.cuda_ORB.create()

    fileloc1 = os.path.join(filepath, template_target)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    cuMat1 = cv2.cuda_GpuMat()
    cuMat1.upload(img1)
    cuMat1 = cv2.cuda.cvtColor(cuMat1,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndComputeAsync(cuMat1, None)


    if type(des1) is not None:

        matchdir = 'matches'

        donepath = os.path.join(basepath, matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.DS_Store' not in file:
                    if '.txt' in file:
                        doneitem = file.replace('.txt','.jpg')
                        done.append(doneitem)

        processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
        processfile = open(processfileloc,"a")
        processfile.close()

        matched_images = []

        for indexb,template_compare in enumerate(templates):
            if template_target != template_compare:
                if template_compare not in done:

                    fileloc2 = os.path.join(filepath, template_compare)
                    img2 = cv2.imread(fileloc2)
                    img2 = imutils.resize(img2, width=800)
                    cuMat2 = cv2.cuda_GpuMat()
                    cuMat2.upload(img2) 
                    cuMat2 = cv2.cuda.cvtColor(cuMat2,cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndComputeAsync(cuMat2, None)

                    if type(des2) is not None:

                        bf = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)

                        matches = bf.match(des1, des2)

                        goodmatches = []

                        for match in matches:
                            if match.distance <= 25:
                                goodmatches.append(match)

                        if len(goodmatches) >= 10:
                            matched_images.append(template_compare)

        if len(matched_images) > 0:
            with open (processfileloc, 'a') as f:
                for item in matched_images:
                    f.write("%s\n" % item)



if __name__ == '__main__':

    start = time.time()

    basepath = os.path.dirname(os.path.realpath(__file__))
    subdir = "source_images_clean"
    filepath = os.path.join(basepath, subdir)

    templatesbase = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templatesbase.append(file)

    # p = Pool()

    # executefunction = p.map(processimgs,templatesbase)

    # p.close()
    # p.join()

    # r = thread_map(processimgs_cpu_flann, templatesbase, max_workers=6)
                

    def sequential(func):
        shutil.rmtree('matches')
        os.mkdir('matches')
        for i,template in enumerate(templatesbase):
            line_prof = LineProfiler()
            line_prof.add_function(func)
            line_prof.runcall(func, template)
            if 'distract' not in template:
                line_prof.print_stats()

    def parallel(func):
        shutil.rmtree('matches')
        os.mkdir('matches')
        thread_map(func, templatesbase)

    setup = """
from __main__ import sequential
from __main__ import parallel
from __main__ import processimgs_cpu_flann_optimized
from __main__ import processimgs_gpu
from __main__ import processimgs_cpu_flann"""
    # Example of using timeit
    execution_time = timeit.timeit("parallel(processimgs_cpu_flann_optimized)",setup=setup, number=10)
    print(f"Average execution time: {execution_time / 10}")

    # sequential(processimgs_cpu_flann_optimized)

    # runtime = time.time()-start
    # print('Script runtime:',round(runtime,2),'seconds',round(runtime/3600,2),'hours')
