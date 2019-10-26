import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.array([[2, 1],
			  [3, 1],
			  [0, 4],
			  [1, 6]])
y = np.array([[-1],
			  [-1],
			   [1],
			   [1]])

S = np.hstack((x, y))

plt.plot(S[:,0], S[:,1], 'ro')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S[:,0], S[:,1], S[:,2], marker='o')
plt.show()

w = [0, 0]
incomplete = True
flag = True
count = 0
iterations = 0


while incomplete:
	while count<4:
		for k in range(0, 4):
			print("loop index = %d" % k)
			print(w)
			print(S[k][0:2])
			print("dot product = %d" % (S[k][2]*np.dot(w,S[k][0:2])))
			if S[k][2]*np.dot(w,S[k][0:2])<=0:
				print("negative or zero value found")
				w += S[k][2]*S[k][0:2]
				iterations +=1
				print("iterations = %d" % iterations)
				#print(incomplete)
				print("count before exiting = %d" % count)
				count = 0
				print("reset count (count = %d)" % count)
				break
			else:
				print("positive value found")
				count +=1
				print("count = %d" % count)
	print("all positive")
	incomplete = False


print("final iteration count = %d" % iterations)
print(w)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(S[:,0], S[:,1], S[:,2], marker='o')

x_plane = np.linspace(-1,7,100)
y_plane = np.linspace(-1,7,100)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = w[0]*X_plane + w[1]*Y_plane
surf = ax.plot_surface(X_plane, Y_plane, Z_plane)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()

for c in range(0, 4):
	print(c)
	print(S[c][0:2])
	print(S[c][2] * np.dot(w, S[c][0:2]))