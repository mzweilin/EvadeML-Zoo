import urllib2
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('datasets/svhn_dataset/')

filename = ['test_32x32.mat','train_32x32.mat']
addr = ['http://ufldl.stanford.edu/housenumbers/test_32x32.mat','http://ufldl.stanford.edu/housenumbers/train_32x32.mat']

for i in range(2):
	f = os.path.join('datasets/svhn_dataset',filename[i])
	if not os.path.exists(f):
		output = open(f,'w')
		output.write(urllib2.urlopen(addr[i]).read())
		output.close()
