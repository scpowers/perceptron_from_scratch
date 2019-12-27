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
# add the label vector to the training data
S = np.hstack((x, y))

# plot the training points in 2D just to sanity check the coordinate assignments
fig1 = plt.plot(S[:, 0], S[:, 1], 'ro')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.title('Training Points')
plt.show(block=False)

# plot the training points in 3D to show that they are, in fact, separable
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(S[:, 0], S[:, 1], S[:, 2], marker='o')
ax.set_title('Training Points with Labels')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Label')
plt.show(block=False)

# begin perceptron implementation

# initialize hyperplane-defining vector
w = [0, 0]
needs_converging = True
iterations = 0
wrong_index = 5
wrong_counter = 0

# main loop that updates the w vector based on misclassified points
while needs_converging:
	print("-----------------------------------------------------------------")
	print("Checking examples sequentially")
	print("iterations = %d" % iterations)
	wrong_counter = 0
	for k in range(0, 4):
		print(w)
		print(S[k][0:2])
		print(S[k][2]*np.dot(w, S[k][0:2]))
		if S[k][2]*np.dot(w, S[k][0:2]) <= 0:
			wrong_index = k
			wrong_counter += 1
			print("Found a misclassified point")
	if wrong_counter == 0:
		print("Converged!")
		needs_converging = False
	else:
		print("Updating based on example %d" % wrong_index)
		w += np.dot(S[wrong_index][2], S[wrong_index][0:2])
		iterations += 1

# return the number of iterations required for convergence
print("final iteration count = %d" % iterations)
# return the final w vector
print(w)

# plot the training data separated by the hyperplane defined by the final w
fig3 = plt.figure()
ax = fig3.gca(projection='3d')
ax.scatter(S[:, 0], S[:, 1], S[:, 2], marker='o')
x_plane = np.linspace(-1, 7, 100)
y_plane = np.linspace(-1, 7, 100)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = w[0]*X_plane + w[1]*Y_plane
surf = ax.plot_surface(X_plane, Y_plane, Z_plane)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Label')
ax.set_title('Training Points with Separating Hyperplane')
plt.show()