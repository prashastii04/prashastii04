
In this project, a case study of a trading strategy is implemented.



There are two csv files attached containing financial time series representing the price of two futures contracts, A and B respectively. 

PART 1
• The following systematic trading strategy is implemented:
- Buys or maintains a long position of size L>0 in futures A every time the price of futures B falls more than X standard deviations (calculated over the previous N days) in a day
- Sells or maintains a short position of size S<0 in futures A every time the price of futures B rises more than Y standard deviations (calculated over the previous N days) in a day
- Closes any position in futures A if the price of futures B is within min(X, Y) standard deviations (calculated over the previous N days) in a day
- The strategy pays costs of C x abs(size) for each position it enters • Produces a chart of cumulative profit and loss of the strategy over time


• Produces a chart of drawdown (absolute loss at time T / maximum achieved profit before T) over time
• Prints the annualised return, Sharpe ratio and max(drawdown) of the strategy where X, Y and N, L, S, C are user defined variables.

It is assumed that:
- The risk-free rate is zero
- You are allowed to take a short position on a futures contract (i.e., the net number of
futures contracts of the same type in instantaneous possession may go negative)

PART 2 

The strategy here is for market-making, and you get paid for entering positions, but you do not get to choose when you trade.
It is assumed that at each timestep you are forcibly entered into a long position of size S with probability p1 and independently a short position of size S with probability p2 (you may simultaneously get entered long and short, in which case your final additional position is 0). In each of these transactions you get paid M x S to be entered into the trades.
