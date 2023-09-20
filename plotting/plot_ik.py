import numpy as np
import subprocess

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from eqns import *
from lgp import init_registers, execute, print_genome

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

x = 0.0
y = 0.0
z = 0.0
th_1 = 0.0
th_2 = 0.0
th_3 = 0.0

# genome = [4, 2, 2, 1, 3, 2, 2, 0]
# genome = [0, 4, 4, 3, 1, 2, 0, 3, 1, 3, 2, 0, 2, 4, 1, 1, 0, 3, 4, 3, 3, 4, 4, 2, 2, 3, 2, 1, 3, 4, 2, 4, 2, 2, 0, 2, 1, 3, 1, 2, 3, 2, 0, 1]
# Best with random angles
# genome = [1, 3, 1, 2, 6, 2, 3, 3, 3, 2, 2, 1, 0, 2, 3, 4, 2, 2, 2, 0, 6, 2, 3, 3, 3, 2, 2, 1, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 3, 0, 3, 3, 3]

genome = [3, 5, 1, 4, 4, 6, 4, 0, 3, 6, 4, 5, 6, 4, 3, 0, 5, 4, 3, 0, 0, 4, 6, 0, 1, 5, 3, 2, 6, 3, 1, 3, 6, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 4, 3, 1, 3, 4, 5, 5, 5, 4, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 6, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 4, 3, 1, 3, 4, 5, 5, 5, 4, 3, 4, 6, 4, 5, 5, 5, 4, 3, 4, 6, 4, 3, 1, 3, 4, 4, 4, 5, 2, 2, 1, 5, 3, 2, 5, 4, 0, 4, 3, 0, 2, 5, 0, 3, 4, 4, 4, 5, 0, 4, 6, 0, 4, 5, 0, 3, 0, 5, 3, 2, 6, 3, 1, 3, 4, 3, 4, 6, 5, 2, 1, 5, 4, 3, 1, 3, 0, 5, 5, 5, 0, 4, 4, 5, 0, 2, 1, 5, 3, 2, 5, 4, 3, 2, 1, 5, 1, 2, 1, 5, 3, 2, 5, 4, 0, 4, 6, 0, 0, 4, 4, 5, 3, 2, 5, 4, 4, 4, 4, 5, 6, 3, 1, 3, 4, 3, 1, 3, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 4, 4, 5, 0, 4, 4, 5, 4, 3, 1, 3, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 4, 3, 1, 3, 1, 4, 4, 5, 1, 2, 1, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 0, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 6, 3, 1, 3, 4, 3, 4, 6, 4, 3, 4, 6, 4, 4, 4, 5, 4, 4, 4, 5, 1, 2, 1, 5, 3, 2, 5, 4, 0, 4, 3, 0, 3, 2, 1, 5, 3, 3, 1, 3, 4, 3, 1, 3, 1, 4, 4, 5, 0, 4, 4, 5, 4, 4, 4, 5, 6, 3, 1, 3, 4, 3, 1, 3, 1, 4, 4, 5, 0, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 6, 3, 1, 3, 4, 3, 1, 3, 1, 4, 4, 5, 0, 4, 4, 5, 1, 4, 4, 5, 4, 4, 4, 5, 0, 4, 4, 5, 4, 3, 1, 3, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 4, 3, 1, 3, 1, 4, 4, 5, 1, 2, 1, 5, 0, 4, 4, 5, 0, 2, 1, 5, 1, 4, 4, 5, 0, 4, 4, 5, 6, 3, 1, 3, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 4, 3, 1, 3, 1, 4, 4, 5, 5, 2, 1, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 0, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 0, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 0, 3, 1, 3, 2, 3, 1, 3, 1, 4, 4, 5, 0, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 0, 4, 4, 5, 1, 2, 1, 5, 3, 2, 5, 4, 1, 2, 1, 5, 1, 5, 3, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 4, 5, 1, 2, 1, 5, 1, 5, 3, 2, 0, 4, 4, 5, 1, 2, 1, 5, 3, 2, 5, 4, 0, 4, 6, 0, 3, 3, 6, 3]

def run_ik():
    n = subprocess.run(["../target/release/ik_lgp", "ik", f"{x}", f"{y}", f"{z}", "../genome.txt"], stdout=subprocess.PIPE, text=True)
    global th_1
    global th_2
    global th_3

    angles = n.stdout.strip().split(" ")
    th_1 = float(angles[0])
    th_2 = float(angles[1])
    th_3 = float(angles[2])

def update1(val):
    global x
    x = val
    run_ik()

    print("update 1")

    draw_hubert(ax)

def update2(val):
    global y
    y = val
    run_ik()
    print("update 1")

    draw_hubert(ax)

def update3(val):
    global z
    z = val
    run_ik()
    print("update 1")

    draw_hubert(ax)

def draw_hubert(ax):
    ax.cla()

    shoulder_pos = A_1_to_0(th_1, np.zeros(3))
    end_of_upper_arm = A_1_to_0(th_1, A_2_to_1(th_2, np.array([-L7, 0.0, L5])))

    elbow_pos = A_1_to_0(th_1, A_2_to_1(th_2, np.zeros(3)))
    hand_pos = A_3_to_0(th_1, th_2, th_3, np.array([0.0, 0.0, 0.0]))

    goal_pos = [x, y, z]

    base_pos = [0, 0, L1]
    neck_pos = [0, 0, L2 + L3]

    ax.plot(*list(zip(base_pos, neck_pos)), "-", c="tab:blue")
    ax.plot(*list(zip(neck_pos, shoulder_pos[:3])), "--", c="tab:blue")
    ax.plot(shoulder_pos[0], shoulder_pos[1], shoulder_pos[2], "b.", label="Shoulder")
    ax.plot(*list(zip(shoulder_pos, end_of_upper_arm))[:3], c="tab:blue")
    ax.plot(elbow_pos[0], elbow_pos[1], elbow_pos[2], "g.", label="Elbow")
    ax.plot(hand_pos[0], hand_pos[1], hand_pos[2], "r.", label="Hand")
    ax.plot(goal_pos[0], goal_pos[1], goal_pos[2], "r.", label="Goal")
    ax.plot(*list(zip(elbow_pos, hand_pos))[:3], c="tab:blue")

    ax.set_xlim(-0.18, 0.40)
    ax.set_ylim(-0.18, 0.40)
    ax.set_zlim(0, 0.7)

    ax.legend([f"x: {x:.2f}, y:{y:.2f}, z:{z:.2f}", f"th_1: {th_1 * 180 / np.pi:.2f} th_2: {th_2 * 180 / np.pi:.2f} th_3: {th_3 * 180 / np.pi:.2f}"])

if __name__ == "__main__":
    fig.subplots_adjust(bottom=0.24)

    th_1_ax = fig.add_axes([0.1, 0.11, 0.8, 0.03])
    th_1_slider = Slider(
        ax=th_1_ax,
        label="$x$",
        valmin=-0.16,
        valmax=0.4,
        valinit=0.1,
        orientation="horizontal"
    )
    th_2_ax = fig.add_axes([0.1, 0.06, 0.8, 0.03])
    th_2_slider = Slider(
        ax=th_2_ax,
        label="$y$",
        valmin=-0.16,
        valmax=0.1,
        valinit=0.1,
        orientation="horizontal"
    )
    th_3_ax = fig.add_axes([0.1, 0.01, 0.8, 0.03])
    th_3_slider = Slider(
        ax=th_3_ax,
        label="$z$",
        valmin=0.06,
        valmax=0.7,
        valinit=0.1,
        orientation="horizontal"
    )

    th_1_slider.on_changed(update1)
    th_2_slider.on_changed(update2)
    th_3_slider.on_changed(update3)

    draw_hubert(ax)

    plt.show()
