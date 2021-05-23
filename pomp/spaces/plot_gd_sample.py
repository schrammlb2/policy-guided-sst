import matplotlib.pyplot as plt

def plot_gd(start, sample_traj, goal):
    # plt.xlim((-1,1))
    # plt.ylim((-1,1))
    traj_x = [pt[0] for pt in sample_traj]
    traj_y = [pt[1] for pt in sample_traj]

    # plt.scatter(traj_x[0], traj_y[1], color='green', s=.1)
    plt.plot(traj_x, traj_y, color='blue')

    circle = plt.Circle((start[0], start[1]), .1, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .1, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)

    plt.title("Trajectory of sampled point")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()