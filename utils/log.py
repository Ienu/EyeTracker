# -*- coding: utf-8 -*-
'''
FileName: log.py
Author: Wenyu
Date: 07/26/2019
Version: v1.0 [07/26/2019][Wenyu] for log
'''

import time

def log(func):
	'''
	decorator for logs
	'''
	def wrapper(*args, **kw):
		local_time = time.localtime(time.time())
		print('Call %s(): at %s' % (func.__name__, time.asctime(local_time)))
		return func(*args, **kw)
	return wrapper
