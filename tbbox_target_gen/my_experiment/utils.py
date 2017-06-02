with open('detection_train_val_25fps_o75_3bg1fg_actionness.txt') as f:
	files = f.readlines()

video = []
for line in files:
	line = line.split(' ')
	video.append(line[0])

print 'video list length:', len(video)
count = 0
for i,item in enumerate(video):
	count += 1
	if video[i+1]!=item:
		print(item + ' ' +str(count)+ ' ' +str(count/4*3))
		count = 0