# change the 1-20 class label to corresponding oringin label [7, 13, ...]

with open('test/class_index_detection.txt') as f:
        class_label = f.readlines()
with open('tmp.txt') as f:
	files = f.readlines()
cls_ind = range(21)
for idx, label in enumerate(class_label):
	cls_ind[idx+1] = label.split(' ')[0]

fout = open('tmp_scnn_10285_sorted.txt','wr')

for line in files:
	item = line.split(' ')
	label = int(item[3])
	item[3] = cls_ind[label]
	# print item
	fout.writelines(item[0] + ' ' + item[1] + ' ' + item[2] +' '+ str(item[3]) +' '+ item[4]+ '\n')
fout.close()
 

