import os

framerate = 25
framedir = '/media/sdy/dataset/dataset/UCF101/newframe/'
videodir = '/media/sdy/workspace1/THUMOS2014/TrainingData/UCF101/'


with open('5class_videolist.txt') as fp:
    lines = fp.readlines()

videoname = range(0, len(lines))

for i in xrange(len(lines)):
    videoname[i] = lines[i].split(' ')[0].strip()
    # os.mkdir(framedir + videoname[i].split('.')[0])

for i in xrange(len(lines)):
    cmd = 'ffmpeg -i ' + videodir + videoname[i] + ' -r ' + str(framerate) + ' -f image2 ' + framedir + videoname[i].split('.')[0] +'/%06d.jpg' + ' 2>' + framedir + 'frame_extract.log'
    os.system(cmd)