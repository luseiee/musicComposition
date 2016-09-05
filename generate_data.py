import sys
for i in range(1000):
	sys.argv = ['read_abc.py', 'dataset/a.abc', str(i+1)]
	try:
		execfile("read_abc.py")
	except:
		print (str(i+1) + 'wrong')