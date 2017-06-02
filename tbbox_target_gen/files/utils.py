with open('ucf_tbbox.txt') as f:
	lines = f.readlines()

fout = open('ucf_tbbox_actionness_list.txt','wr')
for line in lines:
	fout.writelines(line.strip() + ' 1.0\n')
fout.close()
