# Stock-Trading-Reinforcement_Learning
Q-Learning for stock trading, linear approximation of Q function

Command for training: python trader.py -m train && python plot_rewards.py -m train
Command for testing: python trader.py -m test && python plot_rewards.py -m test

Data: 'aapl_msi_sbux.csv', closing price of Apple, Motorola, Starbucks

Enviroment:

	State: shares of the ith stock, price of the ith stock for i in [1,Number_of_stocks] and remaining cash
	
	Actions: sell, hold, buy the ith stock, for i in [1,Number_of_stocks], 3^Number_of_stocks total actions 

Linear Model: Q-function approximation for states' aggregation

Agent: ε-greedy approach to enforce exploration, perfomrs the steps of Q-learning and stochastic gradent decent to update the model's parameters.

Plots: 

      1) histogram_training.png: Values of the episodes based on the training data
      
      2) histogram_testing.png: Values of the episodes based on the testing data
