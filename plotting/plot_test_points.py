import subprocess
import numpy as np

from matplotlib import pyplot as plt

# Define the range and number of steps
start = 0
end = np.pi
num_steps1 = 8  # Adjust the number of steps as needed
num_steps2 = 8  # Adjust the number of steps as needed
num_steps3 = 8  # Adjust the number of steps as needed

points = []

# Loop through the variables
for i in range(num_steps1 + 1):
    th_1 = start + (i / num_steps1) * (end - start)
    for j in range(num_steps2 + 1):
        th_2 = start + (j / num_steps2) * (end - start)
        for k in range(num_steps3 + 1):
            th_3 = start + (k / num_steps3) * (end / 2 - start)

            # Call the external program and capture stdout
            command = ["../target/release/ik_lgp", "fk", str(th_1), str(th_2), str(th_3)]
            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                points.append(list(map(float, output.strip().split(" "))))
            except subprocess.CalledProcessError as e:
                print(f"Error calling external program: {e.returncode}")
                print(e.output)

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(points[:,0], points[:,1], points[:,2], '.')

plt.title("Test points")
plt.xlabel("x")
plt.ylabel("y")
# plt.zlabel("z")

plt.show()
