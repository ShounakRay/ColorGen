############################################
######## KOHONEN SOM-RELATED IMPORTS #######
############################################
import KohonenMap
import GeneratePoints
import NeuralGas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import tkinter
matplotlib.use("TkAgg")
import numpy as np

GeneratePoints.find_points()
SOM_Kohonen = KohonenMap.SelfOrganizingMap(number_of_neurons = 20, input_data_file = "Data/randomPoints.txt", radius = 0.5, alpha = 0.5, gaussian = 0)
SOM_Kohonen.train(20)
# SOM_Kohonen.animate_plots()

fig, ax = plt.subplots()
ax.axis([np.min(SOM_Kohonen.animation_plots[0], axis=0)[0] - 3, np.max(SOM_Kohonen.animation_plots[0], axis=0)[0] + 3,
         np.min(SOM_Kohonen.animation_plots[0], axis=0)[1] - 3, np.max(SOM_Kohonen.animation_plots[0], axis=0)[1] + 3])
ax.plot(SOM_Kohonen.input_data[:, 0], SOM_Kohonen.input_data[:, 1], 'bo')
line, = ax.plot([], [], 'ro')
def animate(frame):
    if frame > len(SOM_Kohonen.animation_plots) - 1:
        frame = len(SOM_Kohonen.animation_plots) - 1
    line.set_data(SOM_Kohonen.animation_plots[frame][:, 0], SOM_Kohonen.animation_plots[frame][:, 1])
    ax.set_title("Input Data: " + str((frame + 1)))
    return line

ani = animation.FuncAnimation(fig, animate, len(SOM_Kohonen.animation_plots), interval=1, repeat=False)
try:
    writer = animation.writers['ffmpeg']
except KeyError:
    writer = animation.writers['avconv']
writer = writer(fps=60)
ani.save('NewMovie.mp4', writer = writer)



GeneratePoints.find_points()
SOM_NeuralGas = NeuralGas.SelfOrganizingMap(number_of_neurons=20, input_data_file="Data/randomPoints.txt", radius=0.5, alpha=0.5)
SOM_NeuralGas.train(20)
