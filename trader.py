import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from Linear_Model import LinearModel
from Stock_Market_Enviroment import StockEnv
from Agent import Agent

from datetime import datetime 
import itertools
import argparse 
import re 
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data():
	df = pd.read_csv('aapl_msi_sbux.csv')
	return df.values

def scaling(env):
	list_of_states= []
	done = False
	while not done:
		action = np.random.choice(env.action_space)
		state, reward, done, info =env.step(action)

		list_of_states.append(state)

	scaler=StandardScaler()
	scaler.fit(list_of_states)
	return scaler
		
def make_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def play_one_episode(agent, env, is_train):

	state = env.reset() # get the initial state 
	state = scaler.transform([state]) # scale the state vector
	done = False

	while not done:
		action = agent.act(state) # get the next action
		next_state, reward, done, info = env.step(action) # perform the action
		next_state = scaler.transform([next_state]) # scale the next state
		if is_train == 'train': # if the mode is training
			agent.train(state, action, reward, next_state, done) # Q-Learning with states' aggregation
		state = next_state # got to next state

	return info['current value']

if __name__=="__main__":
	
	models_folder = 'linear_rl_model' # to store the Q-model prarameters
	rewards_folder = 'linear_rl_rewards' # to store the values of episodes

	num_episodes= 2000 # epochs
	batch_size = 32
	initial_investment = 20000

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')

	args = parser.parse_args()

	make_directory(models_folder)
	make_directory(rewards_folder)

	data= get_data()

	n_timesteps, n_stocks = data.shape

	n_train = n_timesteps // 2 

	train_data = data[:n_train] # half of the data for training
	test_data = data[n_train:] # half of the data for testing


	env = StockEnv(train_data, initial_investment) # initialize the enviroment
	state_size = env.state_dimension # initialize state dimension
	action_size = len(env.action_space)  # initialize actions dimension
	agent = Agent(state_size, action_size) # initialize the agent's class

	scaler = scaling(env) # get the scaling parameters


	portfolio_value = [] # to collect the values at the end of episodes/epochs

	if args.mode == 'test':
		#load previous scaler
		with open(f'{models_folder}/scaler.pkl', 'rb') as f:
			scaler = pickle.load(f)

		# remake the env with the test data
		env = StockEnv(test_data, initial_investment)


		agent.epsilon = 0.01

		agent.load(f'{models_folder}/linear.npz')

	for e in range(num_episodes):
		t0 = datetime.now()
		val = play_one_episode(agent, env, args.mode)
		dt = datetime.now() - t0

		print(f"episode: {e +1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
		portfolio_value.append(val) 

	if args.mode == 'train':

		agent.save(f'{models_folder}/linear.npz')

		with open(f'{models_folder}/scaler.pkl', 'wb') as f:
			pickle.dump(scaler,f)

		plt.plot(agent.model.losses)
		plt.show()
	
	# save portfolio value for each episode
	np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)