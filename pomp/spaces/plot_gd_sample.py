import matplotlib.pyplot as plt

def plot_gd(start, sample_traj, goal):
    assert len(start) == len(goal)
    assert len(sample_traj[0]) == len(goal)
    # plt.xlim((-1,1))
    # plt.ylim((-1,1))
    traj_x = [pt[0] for pt in sample_traj]
    traj_y = [pt[1] for pt in sample_traj]

    circle = plt.Circle((start[0], start[1]), .01, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .01, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)

    # plt.scatter(traj_x[0], traj_y[1], color='green', s=.1)
    # plt.plot(traj_x, traj_y, color='blue')
    for i in range(1, len(sample_traj)):
        x0 = sample_traj[i-1][0]
        y0 = sample_traj[i-1][1]

        x1 = sample_traj[i][0]
        y1 = sample_traj[i][1]

        dx = x1 - x0
        dy = y1 - y0
        plt.arrow(x0, y0, dx, dy, color='blue')


    plt.title("Trajectory of sampled point")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()


def plot_traj(start, sample_traj, goal):
    # assert len(start) == len(goal)
    # assert len(sample_traj[0]) == len(goal)
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    traj_x = [pt[0] for pt in sample_traj]
    traj_y = [pt[1] for pt in sample_traj]

    circle = plt.Circle((start[0], start[1]), .1, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .1, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)

    # plt.scatter(traj_x[0], traj_y[1], color='green', s=.1)
    plt.plot(traj_x, traj_y, color='blue')

    plt.title("Trajectory of sampled point")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()

def plot_roadmap(start, roadmap, min_traj, goal):
    # assert len(start) == len(goal)
    # assert len(sample_traj[0]) == len(goal)
    plt.xlim((-1,1))
    plt.ylim((-1,1))



    # plt.scatter(traj_x[0], traj_y[1], color='green', s=.1)
    for sample_traj in roadmap: 
        traj_x = [pt[0] for pt in sample_traj]
        traj_y = [pt[1] for pt in sample_traj]
        plt.plot(traj_x, traj_y, color='black', linewidth=.1)

    circle = plt.Circle((start[0], start[1]), .1, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .1, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)

    traj_x = [pt[0] for pt in min_traj]
    traj_y = [pt[1] for pt in min_traj]
    plt.plot(traj_x, traj_y, color='blue')

    # plt.title("Trajectory of sampled point")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()



def approximate_vector_field(start, sample_grads, goal):
    assert len(start) == len(goal)
    assert len(sample_grads[0][0]) == len(goal)
    # plt.xlim((-1,1))
    # plt.ylim((-1,1))

    circle = plt.Circle((start[0], start[1]), .05, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .05, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)

    # plt.scatter(traj_x[0], traj_y[1], color='green', s=.1)
    # plt.plot(traj_x, traj_y, color='blue')
    for i in range(0, len(sample_grads)):
        x0 = sample_grads[i][0][0]
        y0 = sample_grads[i][0][1]

        x1 = sample_grads[i][1][0]
        y1 = sample_grads[i][1][1]

        dx = x1 - x0
        dy = y1 - y0
        # plt.arrow(x0, y0, dx, dy, color='blue', head_width=.01)
        plt.scatter(x1, y1, color='blue', s=1)


    plt.title("Trajectory of sampled point")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()

def scatter_value_heatmap(start, sample_vals, goal): 
    assert len(start) == len(goal)
    # assert len(sample_vals[0][0]) == len(goal)
    # plt.xlim((-1,1))
    # plt.ylim((-1,1))
    xs   = [pt[0] for pt in sample_vals]
    ys   = [pt[1] for pt in sample_vals]
    vals = [pt[2] for pt in sample_vals]



    circle = plt.Circle((start[0], start[1]), .01, facecolor='red', edgecolor='red')
    plt.gca().add_patch(circle)
    circle = plt.Circle((goal[0], goal[1]), .01, facecolor='green', edgecolor='green')
    plt.gca().add_patch(circle)



    # plt.plot(xs, ys, 'ok')
    plt.tricontourf(xs, ys, vals)
    
    plt.title("value map")
    # plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
    plt.show()
    plt.close()