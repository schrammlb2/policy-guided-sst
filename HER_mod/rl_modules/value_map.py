import numpy as np
import torch
import matplotlib.pyplot as plt
import os

target_speed = .5
target_vel = [target_speed/2**.5, target_speed/2**.5]

def get_vals(net, target_velocity):
	steps = 10
	vals = np.zeros((steps, steps))
	total = 0
	goal = [0,0]
	goal += target_velocity # velocity goal
	for i in range(steps):
		for j in range(steps):
			loc = [2*i/steps - 1, 2*j/steps - 1] 
			loc += [0,0] # velocity
			# inp = torch.tensor(np.array(loc + [1000] + [0,0]), dtype=torch.float32).unsqueeze(0)
			inp = torch.tensor(np.array(loc + goal), dtype=torch.float32).unsqueeze(0)
			v = net(inp).detach().numpy()
			total += v
			vals[i,j] = v

	return vals

def value_map(net, title="", target_velocity=[0,0]):
	steps = 10
	vals = get_vals(net, target_velocity)
	# vals /= np.abs(total)
	plt.xticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	plt.yticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	heatmap = plt.imshow(vals, interpolation="nearest")
	plt.colorbar(heatmap)
	plt.savefig(os.path.join('results', title + str(target_velocity) + '.png'))
	plt.close()
	# plt.show()

# def velocity_value_map(net, title=""):
# 	target_position = [.2,.2]
# 	agent_position = [-.2,-.2] 
# 	steps = 50

# 	vals = np.zeros((steps, steps))
# 	total = 0
# 	agent_state = agent_position + [0,0]
# 	for i in range(steps):
# 		for j in range(steps):
# 			target_velocity = [2*i/steps - 1, 2*j/steps - 1] 
# 			# inp = torch.tensor(np.array(loc + [1000] + [0,0]), dtype=torch.float32).unsqueeze(0)
# 			inp = torch.tensor(np.array(agent_state + target_position + target_velocity), dtype=torch.float32).unsqueeze(0)
# 			inp2 = torch.tensor(np.array(target_position + target_velocity + [-1,-1,0,0]), dtype=torch.float32).unsqueeze(0)
# 			v = net(inp).detach().numpy() + net(inp2).detach().numpy()
# 			total += v
# 			vals[i,j] = v
# 	# vals /= np.abs(total)
# 	plt.xticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
# 	plt.yticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
# 	heatmap = plt.imshow(vals, interpolation="nearest")
# 	plt.colorbar(heatmap)
# 	plt.savefig(os.path.join('results', title + str(target_velocity) + '.png'))
# 	plt.close()

def velocity_value_map(net, title=""):
	target_position = [.5,.5]
	agent_position = [-.5,-.5] 
	steps = 50

	vals = np.zeros((steps, steps))
	total = 0
	agent_state = agent_position + [0,0]
	for i in range(steps):
		for j in range(steps):
			target_velocity = [2*i/steps - 1, 2*j/steps - 1] 
			# inp = torch.tensor(np.array(loc + [1000] + [0,0]), dtype=torch.float32).unsqueeze(0)
			inp = torch.tensor(np.array(agent_state + target_position + target_velocity), dtype=torch.float32).unsqueeze(0)
			inp2 = torch.tensor(np.array(target_position + target_velocity + [-1,-1,0,0]), dtype=torch.float32).unsqueeze(0)
			v = net(inp).detach().numpy() #+ net(inp2).detach().numpy()
			total += v
			vals[i,j] = v
	# vals /= np.abs(total)
	plt.xticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	plt.yticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	heatmap = plt.imshow(vals, interpolation="nearest")
	plt.colorbar(heatmap)
	plt.savefig(os.path.join('results', title + str(target_velocity) + '.png'))
	plt.close()



def empirical_velocity_value_map(agent, title=""):
	target_position = [.2,.2]
	agent_position = [-.2,-.2] 
	steps = 50

	vals = np.zeros((steps, steps))
	total = 0
	agent_state = agent_position + [0,0]
	for i in range(steps):
		for j in range(steps):
			target_velocity = [2*i/steps - 1, 2*j/steps - 1] 
			# inp = torch.tensor(np.array(loc + [1000] + [0,0]), dtype=torch.float32).unsqueeze(0)
			initial_pos = torch.tensor(np.array(agent_state), dtype=torch.float32)
			loc_path = [torch.tensor(np.array(target_position), dtype=torch.float32)]
			vel_path = [torch.tensor(np.array(target_velocity), dtype=torch.float32)]
			time = agent.run_path(initial_pos, loc_path, vel_path)[0]
			# inp = torch.tensor(np.array(agent_state + target_position + target_velocity), dtype=torch.float32).unsqueeze(0)
			v = -agent.time2q(time)
			total += v
			vals[i,j] = v
	# vals /= np.abs(total)
	plt.xticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	plt.yticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	heatmap = plt.imshow(vals, interpolation="nearest")
	plt.colorbar(heatmap)
	plt.savefig(os.path.join('results', title + '.png'))
	plt.close()


def diff_map(net, title=""):
	steps = 10
	static_vals = get_vals(net, [0,0])
	vel_vals = get_vals(net, target_vel)

	vals = vel_vals - static_vals
	# vals /= np.abs(total)
	plt.xticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	plt.yticks(ticks=np.arange(steps), labels=[str(2*i/steps - 1) for i in range(steps)])
	heatmap = plt.imshow(vals, interpolation="nearest")
	plt.colorbar(heatmap)
	plt.savefig(os.path.join('results', title + '.png'))
	plt.close()

def q_value_map(q_net, actor, title="", target_velocity=[0,0]):
	value_function = lambda state : q_net(state, actor(state))
	value_map(value_function, title, target_velocity=target_velocity)

def q_diff_map(q_net, actor, title=""):
	value_function = lambda state : q_net(state, actor(state))
	diff_map(value_function, title)

def q_velocity_value_map(q_net, actor, title=""):
	value_function = lambda state : q_net(state, actor(state))
	velocity_value_map(value_function, title)