# --------------------------------------
# run multiloss network
# --------------------------------------

import sys, os
import operator
import numpy as np
import numpy.random as npr
import cPickle
import time

def main():
    print 'CAUTION: you need to remove the seg_train.pkl if you want to set new arguments!'
    # generate positive and negative sample
    iou_h = 0.5
    iou_l = 0.1
    framerate = 25
    # argument init
    # zipped = []
    seg_train = []
    # read gt file
    count = 0
    pos_sample_num = 0
    with open('files/val_detection_videolist.txt') as f:
        video_list = f.readlines()
    with open('files/class_index_detection.txt') as f:
        class_index = f.readlines()
    with open('files/temporal_annotation_full.txt') as f:
        temp_ground_truth = f.readlines()
    with open('files/val_detect_uniform16_fps25_overlap75.txt') as f:
        temp_segment = f.readlines()

    frame_dir = '/media/sdy/dataset/dataset/THUMOS14val/frames25/'
    flog = open('log0520_actionness1to5.txt', 'wr')
    cache_file = 'seg_train_actionness_0520.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            seg_train = cPickle.load(fid)
        print 'seg_train loaded from {}'.format(cache_file)
    else:
        for video in video_list:
            count += 1
            print('Processing the {} video.'.format(count))
            seg_box = []
            seg_info = []
            video_name = video.split(' ')[0]
            s_t = time.time()
            # exmaple of current)_gt: [video_name] [label] [start_time] [end_time]
            current_gt = [(video_name, temp_ground_truth[i].split(' ')[1],
                        [temp_ground_truth[i].split(' ')[2],
                        temp_ground_truth[i].split(' ')[3].strip()])
                        for i, row in enumerate(temp_ground_truth)
                        if row.split(' ')[0] == video_name]

            # count step size for gt
            # [gt_video_name, gt_label, gt_box] = zip(*current_gt)

            # start_frm = [float(b[0]) * framerate for b in gt_box]
            # end_frm = [float(b[1]) * framerate for b in gt_box]

            # gt_lens = map(operator.sub, end_frm, start_frm)

            # gt_stepsize = [int(lens) / 16 for lens in gt_lens]

            print 'extract current video gt in ' + str(time.time() - s_t)[0:6] + ' s'

            s_t = time.time()
            for line in temp_segment:

                split = line.split(' ')
                video_dir = split[0]
                video_seg_name = split[0].split('/')[-1]
                class_label = split[2]
                start_frame = split[1]
                step_size = split[3].strip()
                end_frame = float(step_size) * 16 + float(start_frame)
                start_time = float(start_frame) / framerate
                end_time = float(end_frame) / framerate
                if video_seg_name == video_name:
                    seg_info.append([video_dir, start_frame, step_size])
                    seg_box.append([start_time, end_time, class_label])
                else:
                    continue
            print 'extract related segments in current video in ' + str(time.time() - s_t)[0:6] + ' s'
            # if len(zipped) > 0:
            #     [seg_info, seg_box] = zip(*zipped)

            s_t = time.time()
            
            print video_name
            video_tag, target_dx, target_dl, sample_inds, len_fg_bg, actionness = seg_sample(current_gt, seg_box, iou_h, iou_l)
            if len_fg_bg[1] == 0:
                ratio = 0
            else: 
                ratio = float(len_fg_bg[0])/float(len_fg_bg[1])
            flog.writelines(video_name + ': len_bg = {}, len_fg = {}, ratio = {}'.format(len_fg_bg[0], len_fg_bg[1], ratio) + '\n')
            # print 'compute IoU for current segments done in ' + str(time.time() - s_t)[0:6] + ' s'

            # example of seg_info: [video dir] [start frame] [label] [step size] [target_dx] [target_dl ]
            for i in xrange(len(seg_info)):
                # video_dir = seg_info[i][0]
                # start_frame = seg_info[i][1]
                # step_size = seg_info[i][2]
                seg_info[i].insert(2, video_tag[i])
                seg_info[i].append(target_dx[i])   # 4
                seg_info[i].append(target_dl[i])   # 5
                seg_info[i].append(actionness[i])  # 6
                if i in sample_inds:
                    seg_train.append(seg_info[i])

            # add gt
            # seg_info_gt = [[frame_dir + gt_video_name[i], int(start_frm[i]), gt_label[i], gt_stepsize[i], np.zeros(len(gt_label))[i],
            #                np.zeros(len(gt_label))[i]] for i in range(len(gt_label))]
            # seg_train.extend(seg_info_gt)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(seg_train, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote seg_train to {}'.format(cache_file)

    # compute target mean and stds
    # var(x) = E(x^2) - E(x)^2
    flog.close()
    targets = [(seg_train[idx][2], seg_train[idx][4], seg_train[idx][5]) for idx in range(len(seg_train))]
    targets = np.array(targets).astype(np.float32)
    class_counts = np.zeros((21, 1)) + 1e-14
    sums = np.zeros((21, 2))
    squared_sums = np.zeros((21, 2))
    b = targets[:, 0]
    for cls in xrange(1, 21):
        cls_inds = np.where(targets[:, 0] == cls)[0]
        if cls_inds.size > 0:
            class_counts[cls] += cls_inds.size
            sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
            squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'tbbox target means:'
    print means
    print means[1:, :].mean(axis=0)  # ignore bg class
    print 'tbbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0)  # ignore bg class

    means_stds = np.concatenate((means.ravel(), stds.ravel()))

    with open('means_stds_val.pkl', 'wb') as fid:
    	cPickle.dump(means_stds, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote seg_train to {}'.format(cache_file)

    # normalize targets
    print 'normalize targets'
    for cls in xrange(1, 21):
        cls_inds = np.where(targets[:, 0] == cls)[0]
        targets[cls_inds, 1:] -= means[cls, :]
        targets[cls_inds, 1:] /= stds[cls, :]


    fout = open('my_experiment/detection_train_25fps_o75_3bg1fg_actionness1to5_0520.txt', 'w')
    for idx, line in enumerate(seg_train):
        fout.writelines('{0} {1} {2} {3} {4} {5} {6}\n'.format(line[0], line[1], line[2], line[3], targets[idx, 1], targets[idx, 2], line[6]))
    fout.close()





def seg_sample(ground_truth, segment_box, iou_h, iou_l):
    # Calculate the IoU, generate positive sample and negetive sample.
    # Initialize the seg_train matrix for train set using the number below:

    [gt_video_name, gt_label, gt_box] = zip(*ground_truth)

    x1 = [float(b[0]) for b in gt_box]
    x2 = [float(b[1]) for b in gt_box]
    y1 = [b[0] for b in segment_box]
    y2 = [b[1] for b in segment_box]
    sample = [b[2] for b in segment_box]

    o = np.zeros((len(y1), len(x1)))
    a = np.zeros((len(y1), len(x1)))     # actionness
    union_x = map(operator.sub, x2, x1)  # union = x2-x1 of gt
    union_y = map(operator.sub, y2, y1)  # union = y2-y1 of sample

    # temporal bounding box transform
    ctr_x = np.array(x1) + 0.5 * np.array(union_x)  # gt
    ctr_y = np.array(y1) + 0.5 * np.array(union_y)  # example
    # target_dx = (ctr_x - ctr_y) / np.array(union_y)
    # target_dl = np.log(np.array(union_x) / np.array(union_y))

    for i in xrange(len(x1)):  # traverse the gt_box
        xx1 = [max(x1[i], y1[j]) for j in xrange(len(y1))]
        xx2 = [min(x2[i], y2[j]) for j in xrange(len(y2))]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]

        new_o = [inter[u]/(union_x[i] + union_y[u] - inter[u]) for u in range(len(y1))]
        new_o = np.array(new_o)
        o[:, i] = new_o

        new_a = [inter[u]/union_y[u] for u in range(len(y1))]
        new_a = np.array(new_a)
        a[:, i] = new_a

    target_dx = range(len(y1))
    target_dl = range(len(y1))
    actionness = range(len(y1))

    fg_inds = []
    bg_inds = []
    for i in xrange(len(y1)):
        if np.amax(o[i, :]) > iou_h:
            pos = np.argmax(o[i, :], axis=0)
            new_label = gt_label[pos]  # fg
            target_dx[i] = (ctr_x[pos] - ctr_y[i]) / union_y[i]
            target_dl[i] = np.log(union_x[pos] / union_y[i])
            actionness[i] = a[i, pos]
            fg_inds.append(i)
        elif np.amax(o[i, :]) < 0.3:
            pos = np.argmax(o[i, :], axis=0)
            new_label = str(0)  # bg
            if 0.1 < a[i, pos] < iou_h:              
                target_dx[i] = 0
                target_dl[i] = 0
                actionness[i] = a[i, pos]
                bg_inds.append(i)
        else:
            new_label = str(0)  # bg
            target_dx[i] = 0
            target_dl[i] = 0
            actionness[i] = 0
        sample[i] = new_label

    # bg = times * (fg + gt)
    # bg_num = 10 * (len(fg_inds) + len(x1))
    bg_num = 3 * len(fg_inds)
    if len(bg_inds) >= bg_num:
        bg_inds = npr.choice(
                bg_inds, size=bg_num, replace=False)
    len_fg_bg = [len(bg_inds), len(fg_inds)]
    print 'bg_len = {}, fg_len = {}'.format(len_fg_bg[0], len_fg_bg[1])
    fg_inds.extend(bg_inds)
    fg_inds.sort()

    return sample, target_dx, target_dl, fg_inds, len_fg_bg, actionness





if __name__ == "__main__":
    main()
