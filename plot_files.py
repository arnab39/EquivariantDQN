import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.ndimage.filters import gaussian_filter1d

def get_args():
	parser = ArgumentParser(description='Plotting')
	parser.add_argument('--file_dir', type=str, default='./')

	args = parser.parse_args()
	return args

def plot(data, labels):
	models = data.keys()
	fontsize = 24
	fig = plt.figure(figsize=(12, 8))
	plt.rc('font',size=fontsize)
	plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
	plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('legend', fontsize=fontsize)    # legend fontsize
	plt.rc('figure', titlesize=fontsize)
	ax = fig.add_subplot(111)
	
	all_colors = "rbgcmykw"
	colors = {}
	ind = 0
	smoothing_factor = 3

	for key in labels.keys():
		colors[key] = all_colors[ind]
		ind += 1

	models = data.keys()
	models = sorted(models)
	
	for model in models:
		if model not in labels.keys():
			continue
		dat = data[model]
		mean_rewards = gaussian_filter1d(dat['means'], smoothing_factor)
		std_rewards = gaussian_filter1d(dat['std'], smoothing_factor)
		ax.plot(dat['steps'], mean_rewards, label=labels[model], color=colors[model])
		ax.fill_between(dat['steps'], mean_rewards, mean_rewards+std_rewards, alpha=0.4, color=colors[model])
		ax.fill_between(dat['steps'], mean_rewards, mean_rewards-std_rewards, alpha=0.4, color=colors[model])

		
	ax.legend(loc='upper left')
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	#ax.set_title("Avg rewards for various models")
	ax.grid()
	plt.show()


def aggregate_rewards(data):
	models = data.keys()
	print(models)
	mean_vals = {}
	for model in models:
		mod_data = data[model]
		avg = []
		for seed, vals in mod_data.items():
			avg.append(vals['vals'])
		avg = np.array(avg)

		mean_vals[model] = {'steps': vals['steps'], 'means':np.mean(avg, axis=0), 'std':np.std(avg, axis=0)}
	return mean_vals

def compile_file(file_dir, full_file_name):
	# this function assumes that the csv files are in the following format:
	# "run-d4equivariantcnnnormalreplay_seed2-tag-Reward per episode.csv"

	model_name = full_file_name[4:-33]
	seed = full_file_name[-28]

	data_file = pd.read_csv(file_dir + full_file_name)
	steps = data_file.Step.values
	vals = data_file.Value.values

	return seed, steps, vals, model_name

if __name__ == '__main__':
	args = get_args()

	directory = os.path.join(args.file_dir)

	all_data = {}

	for root,dirs,files in os.walk(directory):
		for file in files:
			if file.endswith(".csv"):
				seed, steps, vals, model = compile_file(args.file_dir, file)
				if model not in all_data.keys():
					all_data[model] = {seed: {'steps':steps, 'vals': vals}}
				else:
					all_data[model][seed] = {'steps':steps, 'vals':vals}

	avg_rewards_over_seeds = aggregate_rewards(all_data)

	# plotting equivariant CNN with normal replay vs regular CNN with normal replay
	label_map = {'normal': {'d4equivariantcnnnormalreplay':"D4 equivariant CNN with normal replay", "regularcnnnormalreplay":"Vanilla CNN with normal replay"}, 
				 'priority':{'d4equivariantcnnpriorityreplay': 'D4 equivariant CNN with priority replay', 'regularcnnpriorityreplay': "Vanilla CNN with priority replay"}}

	plot(avg_rewards_over_seeds, label_map['priority'])