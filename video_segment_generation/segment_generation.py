# --------------------------------------------------------------------------
# USAGE EXAMPLE: 
# see example_cmd.txt
# --------------------------------------------------------------------------

import sys, os
from argparse import ArgumentParser

def main():
    print 'extract frame segment starts'
    
    parser = ArgumentParser(description='Extract video temporal proposal.')
    parser.add_argument('-i','--input_list',required=True,
    					help='''Path of the list with video name and class lebel.
    							Eg: /home/sdy/Git/scnn/demo/list.txt''')
    parser.add_argument('-d','--videodir',required=True,
    					help='''The directory of frame folder for each video. 
    							Eg: /media/sdy/dataset/dataset/UCF101/frames/''')
    parser.add_argument('-c','--class_list_dir',required=True,
    					help='''Path to class label list. Format: <class_num> <class_name>
    							Eg: /path/class_index_detection.txt''')
    parser.add_argument('-o','--output_name',required=True,
    					help='''The name of output segment list.Eg: /frame/train_list_uniform16.lst''')
    
    # parse input arguments
    args = parser.parse_args()
    input_list = args.input_list
    videodir = args.videodir
    class_list_dir = args.class_list_dir
    output_name = args.output_name

   	# read the input file
    with open(input_list) as f:
        videolist = f.readlines()

    with open(class_list_dir) as f:
        class_list = f.readlines()

    # ------param init------
    num_video = len(videolist)
    num_class = len(class_list)
    class_index = []
    framerate = 16
    list_total = []
    for i in xrange(num_class):
    	# EG: 7 BasketballPitch
        class_index.append(class_list[i].split(' ')[0])
    for i in xrange(num_video):
    	# EG: video_validateion_0000131 3
        videoname = videolist[i].split(' ')[0]
        frame_path = os.path.join(videodir, videoname)
        num_frame = len(os.listdir(frame_path))
        video_class = videolist[i].split(' ')[1].strip()
        class_label = class_index.index(video_class)
        seg_swin = swin_init(frame_path, framerate, num_frame, class_label)
        list_total.extend(seg_swin)


    # generate proposal list
    # fout1 = open('frame/list_test_proposal.lst', 'w')
    os.mknod(output_name)
    fout2 = open(output_name, 'w')

    for i in range(len(list_total)):
        # fout1.write('/output/'+'{0:06}'.format(i+1)+'\n')
        fout2.write(list_total[i][0] + ' ' + str(list_total[i][2])+' ' + str(list_total[i][6]) + ' '+ str(list_total[i][1]/16) + '\n')
    # fout1.close()
    fout2.close()




def swin_init(videodir, framerate, num_frame, class_label):
    # framerate is the number of frames you want to extract each second from current video.
    # Initial the seg_swin matrix for this video using the numbers below:
    # 1:video_path 2:frame_size_type 3:start_frame 4:end_frame 5:start_time 6:end_time 12:win_overlap_rate
    # seg_swin is a matrix with 12 columns and n rows, where n is the number of segments
    win_overlap_rate = 0.75
    seg_swin = []
    linenum = 0
    for window_stride in [16,32,64,128,256,512]:
        win_overlap = int(window_stride*(1-win_overlap_rate))
        start_frame = 1
        end_frame = window_stride
        while end_frame <= num_frame:
            seg_swin.append([0]*12) # a list of zeros
            seg_swin[linenum][0] = videodir
            seg_swin[linenum][1] = window_stride
            seg_swin[linenum][2] = start_frame
            seg_swin[linenum][3] = end_frame
            seg_swin[linenum][4] = float(start_frame)/framerate
            seg_swin[linenum][5] = float(end_frame)/framerate
            seg_swin[linenum][6] = class_label
            seg_swin[linenum][11] = win_overlap_rate
            # prepare for next iteration
            linenum = linenum+1
            start_frame = start_frame + win_overlap
            end_frame = end_frame + win_overlap
    return seg_swin

if __name__ == "__main__":
    main()
