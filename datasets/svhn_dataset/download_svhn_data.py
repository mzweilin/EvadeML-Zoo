import urllib2
import os

filename = ['test_32x32.mat','train_32x32.mat']
addr = ['http://ufldl.stanford.edu/housenumbers/test_32x32.mat','http://ufldl.stanford.edu/housenumbers/train_32x32.mat']

for i in range(2):
	if not os.path.exists(filename[i]):
		output = open(filename[i],'w')
		output.write(urllib2.urlopen(addr[i]).read())
		output.close()
