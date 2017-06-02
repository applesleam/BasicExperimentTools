import csv
import operator
def main():
    with open('seg_swin_tbbox_0515_weight_only.csv', 'rb') as f:
        file = csv.reader(f)
        seg_swin = [row for row in file]
    with open('test/class_index_detection.txt') as f:
        class_label = f.readlines()
    with open('test/test_detection_videolist.txt') as f:
        list_ = f.readlines()

    cls_ind = range(21)
    for idx, label in enumerate(class_label):
        cls_ind[idx+1] = label.split(' ')[0]

    video_list = list()
    for i in range(len(list_)):
        video_list.append(list_[i].split(' ')[0])

    # nms
    overlap_nms = 0.3
    new_seg_swin = list()
    cur = seg_swin[0]
    for video in video_list:
        seg_swin_per_video = [row for row in seg_swin if row[0] == video]
        pick_nms = []
        new_seg_swin_per_video = nms(seg_swin_per_video, pick_nms, overlap_nms, cls_ind)
        new_seg_swin.extend(new_seg_swin_per_video)

    seg_swin = new_seg_swin

    myfile = open('result/seg_swin_tbbox_nms3_once_0516_weight.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(seg_swin)
    myfile.close()

    res_out = open('result/res_out_eval_nms3_once_0516_weight.txt', 'w')
    # submission format
    #  [video_name] [start_time] [end_time] [class_label] [confidence_score]
    for idx in range(len(seg_swin)):
        if seg_swin[idx][7] != 0:
            # find the origin label in 101
            res_out.writelines(seg_swin[idx][0] + ' ' + str("%.1f"%float(seg_swin[idx][10])) + ' ' + str("%.1f"%float(seg_swin[idx][11])) + ' ' + seg_swin[idx][7] + ' ' + str(seg_swin[idx][6]) + '\n')
    res_out.close()


def nms(seg_swin_per_video, pick_nms, overlap_nms, cls_ind):
    for cls in range(20):
        zipped = [(idx, [seg_swin_per_video[idx][10], seg_swin_per_video[idx][11], seg_swin_per_video[idx][6]]) for idx, row in enumerate(seg_swin_per_video) if int(cls_ind.index(row[7]))-1 ==cls]
        if len(zipped) > 0:
            [inputpick, valuepick] = zip(*zipped)
        else:
            continue
        pick_nms.extend([inputpick[idx] for idx in nms_temporal(valuepick, overlap_nms)])
    new_seg_swin = []
    new_seg_swin = [seg_swin_per_video[idx] for idx in pick_nms]
    return new_seg_swin


def nms_temporal(tbboxes, overlap):
    pick = []

    if len(tbboxes) == 0:
        return pick

    x1 = [float(b[0]) for b in tbboxes]
    x2 = [float(b[1]) for b in tbboxes]
    s = [float(b[-1]) for b in tbboxes]

    union = map(operator.sub, x2, x1)
    # sort the score in increasing seq and get index
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])]

    while len(I) > 0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I) -1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <= overlap:
                I_new.append(I[j])
        I = I_new
    return pick

if __name__ == "__main__":
    main()
