with open('detection_train_25fps_o75_3bg1fg_ucfandval.txt') as f:
	list = f.readlines()

fout = open('detection_train_25fps_o75_10bg1fg_ucfval_classification.txt', 'wr')
for line in list:
	videodir = line.split(' ')[0]
	startfrm = line.split(' ')[1]
	classlabel = line.split(' ')[2]
	stepsize = line.split(' ')[3]
	fout.writelines(videodir + ' ' + startfrm + ' '  + classlabel + ' ' + stepsize + '\n')
fout.close()
