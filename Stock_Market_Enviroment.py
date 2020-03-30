import numpy as np 
import pandas as pd 
import itertools

class StockEnv:
	'''
	Enviroment with Number_of_stocks total stocks
	State: shares of the ith stock, price of the ith stock for i in [1,Number_of_stocks] and remaining cash
	Actions: sell, hold, buy the ith stock, for i in [1,Number_of_stocks], 3^i total actions

	This project considers three stocks (Number_of_stocks): Apple, Motorola, Starbucks 
	'''
	def __init__(self,data,initial_investment):

		#data
		self.stock_price_history = data
		self.num_samples, self.num_stocks =  self.stock_price_history.shape

		#attributes
		self.initial_investment=initial_investment
		self.curr_step = None
		self.stock_owen = None
		self.stock_price = None
		self.cash_in_hand = None

		self.action_space = np.arange(3**self.num_stocks) # 3 actions sell, hold, buy
		self.action_list = list(map(list, itertools.product(np.arange(self.num_stocks), repeat=self.num_stocks)))

		self.state_dimension = 2 * self.num_stocks + 1

		self.reset()

	def reset(self):
		self.curr_step = 0
		self.stock_owned = np.zeros(self.num_stocks)
		self.stock_price = self.stock_price_history[self.curr_step]
		self.cash_in_hand = self.initial_investment
		return self.get_obs()

	def step(self,action):
		assert action in self.action_space # return assertion error if there is no such an action

		previous_value = self.get_value() # get the value before the action

		 
		self.stock_price = self.stock_price_history[self.curr_step] # update prices using the next sample

		# perform the action
		self.trade(action)
		self.curr_step +=1 # go to the next day
		self.stock_price = self.stock_price_history[self.curr_step] # update prices using the next sample
		# get the new value
		curr_value=self.get_value()

		# find the reward
		reward = curr_value - previous_value

		# End of the data
		done = (self.curr_step == self.num_samples-1)

		info = {'current value': curr_value}

		return self.get_obs(), reward, done, info

	def get_obs(self):
		obs = np.empty(self.state_dimension)
		obs[0:self.num_stocks] = self.stock_owned
		obs[self.num_stocks:2*self.num_stocks] = self.stock_price
		obs[-1] = self.cash_in_hand
		return obs

	def get_value(self):
		return self.stock_owned.dot(self.stock_price) + self.cash_in_hand 


	def trade(self,action):

		# example of action for 3 stock: action = [1, 0 ,2]
		# 0: sell
		# 1: hold
		# 2: buy
		action_vector = self.action_list[action]
		# first we sell all that we want to sell to create available cash and then we buy
		sell_index = []
		buy_index = []
		for i in range(len(action_vector)):
			if action_vector[i]==0:
				sell_index.append(i)
			elif action_vector[i]==2:
				buy_index.append(i)

		# first sell
		if sell_index:
			for i in sell_index:
				self.cash_in_hand = self.cash_in_hand + self.stock_owned[i] * self.stock_price[i]
				self.stock_owned[i] = 0 #sell them all

		if buy_index:
			can_buy = True  
			while can_buy:
				can_buy = False
				for i in buy_index:
					if self.cash_in_hand > self.stock_price[i]:
						self.cash_in_hand = self.cash_in_hand - self.stock_price[i]
						self.stock_owned[i] += 1

						if self.cash_in_hand > self.stock_price[i]:
							can_buy = True