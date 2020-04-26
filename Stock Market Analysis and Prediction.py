#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


# In[86]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from pandas_datareader import DataReader # read data from yaahoo

from datetime import datetime 


# In[9]:


from __future__ import division # for division


# In[11]:


# list of texh_stocks for analtytics
tech_list = ['AAPL','GOOGL','MSFT','AMZN']

#set up start and end time for data grab
end = datetime.now()
start = datetime(end.year -1, end.month, end.day) # 1 year back date
print("End time ",end,"Start time ", start)

# for loop for grabbing data and setting as a dataframe
#dataframe as stock ticker
# pip install fix-yahoo-finance (google doesn't work try yahoo)
#globals() function in Python returns the dictionary of current global symbol table.
for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo' , start, end)


# In[18]:


AAPL.head()


# In[16]:


AAPL.describe()


# In[17]:


AAPL.info()


# In[28]:


#plot is the DataFrame func in pandas
AAPL['Close'].plot(legend = True, figsize = (10,4))


# In[35]:


AAPL['Volume'].plot(legend = True, figsize= (10,5)) 


# In[42]:


MA_day = [10,20,50,100] 
# pandas rolling mean calculator?
for ma in MA_day:
    column_name = 'MA for %s days'%(str(ma))
    AAPL[column_name] = AAPL['Close'].rolling(ma).mean()


# In[51]:


AAPL[['Close','MA for 10 days','MA for 20 days','MA for 50 days','MA for 100 days']].plot(subplots = False, figsize = (11,4))


# Daily REturn Analysis

# In[57]:


#pct_changes find the percentage with the previos row
AAPL['Daily Return']=AAPL['Close'].pct_change()


# In[65]:


AAPL['Daily Return'].plot(marker = '*', figsize=(10,7), linestyle = ':')


# In[88]:


AAPL['Daily Return'].plot(kind = 'hist', bins=100)


# In[74]:


#find missing value count
AAPL['Daily Return'].shape[0] - AAPL['Daily Return'].dropna().shape[0]


# In[87]:


#
sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color ='magenta')


# In[90]:


closing_price = DataReader(tech_list, 'yahoo',start,end)['Close']


# In[91]:


closing_price.head()


# In[92]:


closing_price_changes = closing_price.pct_change()


# In[93]:


closing_price_changes.head()


# In[95]:


sns.jointplot('GOOGL','GOOGL',closing_price_changes)


# we can see that if two stock are positive corelated with each other a linear relationship between its daily return value 

# In[98]:


sns.jointplot('AAPL','GOOGL',closing_price_changes, kind = 'scatter', height = 8, color = 'skyblue')


# In[103]:


# with hex plot
sns.jointplot('AMZN','GOOGL',closing_price_changes, kind = 'hex', height = 10)


# In[104]:


sns.jointplot('MSFT','GOOGL',closing_price_changes, kind = 'kde')


# In[120]:


sns.jointplot('AAPL','MSFT',closing_price_changes, kind= 'reg', color = 'skyblue')


# In[ ]:


from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')


# seaborn and pandas make it very easy to repeat this comparison analysis for every possible combination. we can use sns.pairplot to automatically create this plot

# In[119]:


sns.pairplot(closing_price_changes.dropna(), height =2)


# Above we can see all the realationship between all the stocks. A quick glance show an interesting corelation between google and Amazin daily return. sns.pairplot is the fantastic way to display the result. We can make use of pairgrid for full control of the figure

# In[126]:


return_fig =sns.PairGrid(closing_price_changes.dropna())
#defune the upper triangle
return_fig.map_upper(plt.scatter, color='purple')
#define the lower triangle
return_fig.map_lower(sns.kdeplot, color='skyblue')
#finally we define the diagonal as a series of histogram
return_fig.map_diag(plt.hist, bins= 30)


# In[132]:


returns_fig1 = sns.PairGrid(closing_price.dropna())
returns_fig1.map_lower(plt.scatter, color='purple')
returns_fig1.map_upper(sns.kdeplot, cmap = 'cool_d')
returns_fig1.map_diag(plt.hist, bins = 30)


# We can get the corelation plot of the closing_price

# In[138]:


closing_price_changes.corr()


# In[143]:


# Let's go ahead and use seaborn for a quick heatmap to get correlation for the daily return of the stocks.
# annot=> to wite the value inside the box
#fmt = to show the right corner line 
sns.heatmap(closing_price_changes.corr(), annot= True, fmt = ".3g", cmap= 'YlGnBu')


# In[144]:


# check the same for closing_price
sns.heatmap(closing_price.corr(), annot= True, fmt = ".3g", cmap= 'YlGnBu')


# we can visualize the strong corelation between daily stock return
# 

# # Risk Analysis

# There are many ways we can quantify risk, one of the most basic ways using the information we've gathered on daily percentage returns is by comparing the expected return with the standard deviation of the daily returns(Risk).

# In[151]:


rets = closing_price_changes.dropna()


# In[159]:


rets.head()
rets.mean()
rets.std()
rets.columns


# In[167]:


area = np.pi * 30
plt.scatter(rets.mean(),rets.std(), s=area)
plt.xlim([-0.0005,0.0030])
plt.ylim([0.001,0.050])

#Set the plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label,x,y in zip(rets.columns, rets.mean(), rets.std()):
    print(label,x,y)
    plt.annotate(label,
                xy = (x,y), xytext = (50,50),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                arrowprops = dict(arrowstyle = 'fancy', connectionstyle = 'arc3,rad=-0.3'))


# From the above graph it is clear that the lower risk and the positive expected returns
# 
# 
# Value at Risk
# 
# Let's go ahead and define a value at risk parameter for our stocks. We can treat value at risk as the amount of money we could expect to lose (aka putting at risk) for a given confidence interval. There's several methods we can use for estimating a value at risk. Let's go ahead and see some of them in action.
# 
# Value at risk using the "bootstrap" method
# For this method we will calculate the empirical quantiles from a histogram of daily returns. For more information on quantiles, check out this link: http://en.wikipedia.org/wiki/Quantile
# 
# Let's go ahead and repeat the daily returns histogram for Apple stock.

# In[168]:


AAPL.head()


# # make use of dropna to drop the NAN values
# sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'orange')

# In[185]:


rets['AAPL'].quantile(0.05)


# The 0.05 empirical quantile of daily returns is at -0.034. That means that with 95% confidence, our worst daily loss will not exceed 3.4%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.034 * 1,000,000 = $34,000.

# Now we will calculate the quantile for all the stock

# In[186]:


rets['AMZN'].quantile(0.05) # here 2.7 percent


# In[187]:


rets['GOOGL'].quantile(0.06) # here 3.1 percent


# In[188]:


rets['MSFT'].quantile(0.06)


# Value at Risk using the Monte Carlo method
# 
# Using the Monte Carlo to run many trials with random market conditions, then we'll calculate portfolio losses for each trial. After this, we'll use the aggregation of all these simulations to establish how risky the stock is.
# 
# Let's start with a brief explanation of what we're going to do:
# 
# We will use the geometric Brownian motion (GBM), which is technically known as a Markov process. This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of the efficient market hypothesis (EMH): past price information is already incorporated and the next price movement is "conditionally independent" of past price movements.
# 
# This means that the past information on the price of a stock is independent of where the stock price will be in the future, basically meaning, you can't perfectly predict the future solely based on the previous price of a stock.

# Now we see that the change in the stock price is the current stock price multiplied by two terms. The first term is known as "drift", which is the average daily return multiplied by the change of time. The second term is known as "shock", for each time period the stock will "drift" and then experience a "shock" which will randomly push the stock price up or down. By simulating this series of steps of drift and shock thousands of times, we can begin to do a simulation of where we might expect the stock price to be.
# 
# For more info on the Monte Carlo method for stocks and simulating stock prices with GBM model ie. geometric Brownian motion (GBM).
# 
# check out the following link: http://www.investopedia.com/articles/07/montecarlo.asp
# 
# To demonstrate a basic Monte Carlo method, we will start with just a few simulations. First we'll define the variables we'll be using in the Google stock DataFrame GOOGL

# In[192]:


# set our time horizon
days = 365

#now our delta
dt = 1/days

# Now let's grab our mu (drift) from the expected return data we got for GOOGL
mu = rets.mean()['GOOGL']

# Now let's grab the volatility of the stock from the std() of the average return for GOOGL
sigma = rets.std()['GOOGL']


# In[196]:


def stock_monte_carlo(start_price, days,mu,sigma):
    price = np.zeros(days)
    price[0]= start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        
        #calculate Shock
        shock[x] = np.random.normal(loc = mu * dt, scale = sigma * np.sqrt(dt))
        #Calculate Drift
        drift[x] = mu * dt
        #calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
    
    return price


# In[194]:


GOOGL.head()


# In[197]:


start_price= 1272.80

for i in range(100):
    plt.plot(stock_monte_carlo(start_price, days,mu,sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monto Carlo Analysis for Google")


# In[198]:


AMZN.head()


# In[201]:


start_price =1922.45

for i in range(100):
    plt.plot(stock_monte_carlo(start_price, days,mu,sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monto Carlo Amazon Analysis")


# In[202]:


MSFT.head()


# In[203]:


# for microsoft
start_price = 131.37

for i in range(100):
    plt.plot(stock_monte_carlo(start_price, days,mu,sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monto Carlo Analysis For Microsoft")


# Let's go ahead and get a histogram of the end results for a much larger run. (note: This could take a little while to run , depending on the number of runs chosen)

# In[205]:


start_price = 1272.80

runs = 10000
simulation = np.zeros(runs)

for i in range(10000):
    simulation[i] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# Now that we have our array of simulations, we can go ahead and plot a histogram ,as well as use qunatile to define our risk for this stock.
# 
# For more info on quantiles, check out this link: http://en.wikipedia.org/wiki/Quantile

# In[221]:


# Now we'll define q as the 1% empirical quantile, this basically means that 99% of the values should fall between here
q = np.percentile(simulation,1)
plt.hist(simulation, bins = 200)
# Using plt.figtext to fill in some additional information onto the plot

#Starting price
plt.figtext(0.6,0.8, s='Start Price : $%.2f'%start_price)

# mean ending price
plt.figtext(0.6,0.7, s='Mean Final Price: $%.2f' % simulation.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6,0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# To display 1% quantile
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# For plot title
plt.title(label="Final price distribution for Google Stock(GOOGL) after %s days" % days, weight='bold', color='Y')


# Awesome! Now we have looked at the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Google Stock(GOOGL), which looks to be $63.84 for every investment of 1272.80 (The price of one initial Google Stock).
# 
# This basically means for every initial GOOGL stock you purchase you're putting about $63.84 at risk 99% of the time from our Monte Carlo Simulation.
# 
# Now lets plot remaining Stocks to estimate the VaR with our Monte Carlo Simulation.

# In[233]:


start_price = 1922.45

runs = 10000
simulation = np.zeros(runs)

for i in range(10000):
    simulation[i] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[234]:


# Now we'll define q as the 1% empirical quantile, this basically means that 99% of the values should fall between here
q = np.percentile(simulation,1)
plt.hist(simulation, bins = 200)
# Using plt.figtext to fill in some additional information onto the plot

#Starting price
plt.figtext(0.6,0.8, s='Start Price : $%.2f'%start_price)

# mean ending price
plt.figtext(0.6,0.7, s='Mean Final Price: $%.2f' % simulation.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6,0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# To display 1% quantile
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# For plot title
plt.title(label="Final price distribution for Google Stock(AMZN) after %s days" % days, weight='bold', color='Y')


# Now lets estiamte the Value at Risk(VaR) for a stock related to other domains.
# 
# We'll estimate the VaR for:
# 
# Johnson & Johnson > JNJ (U.S.: NYSE) JNJ
# 
# Wal-Mart Stores Inc. > WMT (U.S.: NYSE) WMT
# 
# Nike Inc. > NKE (U.S.: NYSE) NKE
# 
# By using the above methods to get Value at Risk.

# In[237]:


nyse_list = ['JNJ','WMT','NKE']

end_date = datetime.now()
start_date = datetime(end_date.year-1, end_date.month, end_date.day)
print(start_date, end_date)


for i in nyse_list:
    globals()[i] = DataReader(i,'yahoo', start_date, end_date)


# In[239]:


JNJ.describe()


# In[240]:


JNJ.head()


# In[246]:


#legend =
JNJ['Close'].plot(title = 'Closing Price',x = 'Month',y = 'Price',legend = True, figsize= (10,4))


# In[247]:


# Let's see a historical view of the closing price for WMT(Wal-Mart Stores Inc.)
WMT.plot()


# In[253]:


# Let's see a historical view of the closing price for WMT(Wal-Mart Stores Inc.)
WMT['Close'].plot(kind = 'hist', legend= True, figsize = (10,5), title ="Walmart closing")


# In[254]:


WMT['Close'].plot(legend= True, figsize = (10,5), title ="Walmart closing")


# In[257]:


JNJ['Daily Return'] = JNJ['Close'].pct_change()


# In[258]:


JNJ.head()


# In[260]:


#bins => seperation
sns.distplot(JNJ['Close'].dropna(), bins = 100)


# In[261]:


JNJ['Daily Return'].dropna().quantile(0.05)


# The 0.05 empirical quantile of JNJ stock daily returns is at -0.010. That means that with 95% confidence, our worst daily loss will not exceed 1%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.029 * 100= $29.

# In[266]:


WMT['Daily Return'] = WMT['Close'].pct_change()


# In[267]:


WMT.head()


# In[269]:


WMT['Daily Return'].dropna().quantile(0.05)


# The 0.05 empirical quantile of WMT stock daily returns is at -0.013. That means that with 95% confidence, our worst daily loss will not exceed 1.9%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.019 * 1,000,000 = $19,000.

# In[272]:


# use lower case in color bcozz uppercase is deprected
sns.distplot(WMT['Daily Return'].dropna(), bins = 100, color = 'r')


# In[279]:


# Repeat for Nike
NKE['Daily Return'] = NKE['Close'].pct_change()


# In[277]:


NKE['Close'].plot(legend = True, figsize= (10,5))


# In[280]:


NKE.head()


# In[284]:


sns.distplot(NKE['Daily Return'].dropna(), bins = 100, color = 'b')


# In[285]:


NKE['Daily Return'].dropna().quantile(0.05)


# The 0.05 empirical quantile of NKE stock daily returns is at -0.032. That means that with 95% confidence, our worst daily loss will not exceed 3.2%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.032 * 1,000,000 = $32,000.

# In[ ]:




