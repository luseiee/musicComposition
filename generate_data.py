import sys
for i in range(2000):
	sys.argv = ['read_abc.py', 'dataset/mixed.abc', str(i+1)]
	try:
		execfile("read_abc.py")
	except:
		pass