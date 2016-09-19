import sys
for i in range(959):
	sys.argv = ['read_abc.py', 'dataset/jazz.abc', str(i+1)]
	try:
		execfile("read_abc.py")
	except:
		pass