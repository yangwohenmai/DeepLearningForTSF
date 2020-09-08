import numpy as np
a = np.arange(1, 7).reshape((2, 3))
b = np.arange(7, 13).reshape((2, 3))
c = np.arange(13, 19).reshape((2, 3))

print('a = \n', a)
print('b = \n', b)
print('c = \n', c)


s = np.vstack((a, b, c))
print('vstack \n ', s.shape, '\n', s)


s = np.hstack((a, b, c))
print('hstack \n ', s.shape, '\n', s)



s = np.stack((a, b, c), axis=0)
print('axis = 0 \n ', s.shape, '\n', s)


s = np.stack((a, b, c), axis=1)
print('axis = 1 \n ', s.shape, '\n', s)


s = np.stack((a, b, c), axis=2)
print('axis = 2 \n ', s.shape, '\n', s)


#a = np.array([[1,2,3],[4,5,6]])
#b = np.array([[ 7,8,9],[10,11,12]])
#c = np.array([[13,14,15],[16,17,18]])
#print(np.dstack((a,b,c)))


a = np.arange(1, 7).reshape((2, 3))
b = np.arange(7, 13).reshape((2, 3))
c = np.arange(13, 19).reshape((2, 3))
d = np.arange(19, 25).reshape((2, 3))


s = np.stack((a, b, c,d), axis=2)
print('axis = 2 \n ', s.shape, '\n', s)

s = np.dstack((a, b, c,d))
print('axis = 3 \n ', s.shape, '\n', s)