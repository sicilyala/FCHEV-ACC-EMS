import numpy as np
import random
from collections import deque		# 两端都可以操作的序列


class MemoryBuffer:
	def __init__(self, args):
		self.maxSize = int(args.buffer_size)
		self.buffer = deque(maxlen=self.maxSize)
		self.currentSize = 0
		self.counter = 0
		self.batchSize = int(args.batch_size)

	def sample(self):
		"""
		samples a random batch from the replay memory buffer
		"""
		batch = random.sample(self.buffer, self.batchSize)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s_next_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s_next_arr

	def add(self, s, a, r, s_next):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s_next: next state
		:return:
		"""
		transition = (s, a, r, s_next)
		self.buffer.append(transition)
		self.counter += 1
		self.currentSize = min(self.counter, self.maxSize)
