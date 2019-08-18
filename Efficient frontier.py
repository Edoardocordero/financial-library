# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:38:53 2019

@author: Edoardo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from bond import bond
fh = open('Art efficiency.txt', 'w')

############## FRONTIERA EFFICIENTE ##########################

# Lista con i soli titoli presenti sul mercato
tlist = ['^GSPC', 'IYR', '^DJI', '^HUI', '^N225'] 

# Raccolto i dati in un dataframe (prezzi aggiustati)
data_price = pd.DataFrame()
for t in tlist:
    data_price[t] = web.DataReader(t, 'yahoo', '2010-1-1', '2019-04-22')\
    ['Adj Close']
data_price['Bond'] = bond
    
# Calcolo i rendimenti (continous compounding)    
data_ret = np.log(data_price/data_price.shift(1))
    
pfolio_ret = []
pfolio_vol = []
weights_first = []
weights_second = []
weights_third = []
weights_fourth = []
weights_fifth = []
bond_weight = []

weights = []

#Genero randomicamente i pesi dei titoli e calcolo il rendimento medio e la
#volatilità del relativo portafogli salvando i risultati in liste
for i in range(30000):
    weights = np.random.random(len(tlist) + 1)
    
# Standardizzo i pesi così che la somma sia 1
    weights /= sum(weights)
# Formula del ritorno di portafogli
    pfolio_ret.append(np.dot(data_ret.mean()*250, weights))
# Formula della deviazione standard di portafogli
    pfolio_vol.append((np.dot(weights.T, \
                       np.dot(data_ret.cov()*250, weights)))**0.5)
# Liste con i pesi dei primi due titoli
    weights_first.append(weights[0])
    weights_second.append(weights[1])    
    weights_third.append(weights[2])
    weights_fourth.append(weights[3])
    weights_fifth.append(weights[4])
    bond_weight.append(weights[5])
    
# Modifico il tipo lista in np.array per poter creare il dataframe
pfolio_ret = np.array(pfolio_ret)
pfolio_vol = np.array(pfolio_vol)
weights_firts = np.array(weights_first)
weights_second = np.array(weights_second)
weights_third = np.array(weights_third)
weights_fourth = np.array(weights_fourth)
weights_fifth = np.array(weights_fifth)
bond_weight = np.array(bond_weight)
    
# Creo la matrice con i dati dei portafogli, ritorno atteso volatilità e i 
# pesi dei primi due titoli
portfolio = pd.DataFrame({'Exp Return': pfolio_ret,\
                        'Volatility': pfolio_vol, \
                        tlist[0]+\
                        ' weight': weights_first, tlist[1]+' weight': \
                         weights_second, tlist[2]+ ' weight': \
                         weights_third, tlist[3]+' weight': weights_fourth,\
                         tlist[4] +' weight': weights_fifth, ' Bond weight':\
                         bond_weight})
    
# Mostro solo le prime 4 righe della matrice
print(portfolio.head(4))
    
# Plot della frontiera efficiente con uno scatter plot
portfolio.plot(x='Volatility', y='Exp Return', kind= 'scatter', \
               figsize= (15, 6))
plt.show()

#Rendimento e volatilità (0) del tasso risk free
coeff = []
x_free = 0
y_free = 0.01

############# RENDIMENTO RISK FREE E MODIFICA FRONTIERA   ##################

# Calcolo il coefficiente angolare della retta che collega il tasso risk free
# e tutti possibili portafogli. Append poi tuti i coefficienti nella lista coeff
for i in range(30000):
    coeff.append((y_free  - portfolio.iloc[i, 0])\
                 /(x_free - portfolio.iloc[i, 1]))
    
# Trovo il coefficiente angolare massimo della retta che collega quindi il tasso
# risk free al portafogli tangente (di mercato)    
coeff_max = max(coeff)

# Trovo l'indice relativo al portafogli di mercato
idx = coeff.index(max(coeff))

# Mostro i pesi del portafogli di mercato
market_portfolio_weights = []
for i in range(len(tlist) + 1):
    market_portfolio_weights.append(portfolio.iloc[idx, 2+i])

print('\nMarket portfolio weights:\n' + tlist[0] \
      + ' : '+str(round(market_portfolio_weights[0],3))+ '\n' \
      + tlist[1]+' : '+str(round(market_portfolio_weights[1],3))+ \
      '\n' + tlist[2]+ ':' + str(round(market_portfolio_weights[2],3)))
print('\nMarket portoflio Index in DataFrame:', idx)   
 
# Con portfolio.loc[idx] vado a trovare la riga con tutti i dati del portafogli
# di mercato
print('\n\nMkt Portfolio:\n',portfolio.loc[idx])
x1, y1 = [0, 0.3], [y_free, coeff_max*0.3 + y_free]
x2, y2 = [0, 0.3], [y_free, coeff_max*0.3 + y_free]

cal = str(y_free) + ' + Beta (i) x ' + str(portfolio.iloc[idx,0] - y_free)

portfolio.plot(x='Volatility', y='Exp Return', kind= 'scatter', \
               figsize= (15, 6))
plt.xlabel('Standard Deviation')
plt.text(0, 0.15,'Capital Allocation Line:\n' + cal, fontsize=14)
plt.plot(x1, y1, x2, y2, marker='o')
plt.show()

print('\n\nSecurity Market Line:\nE[rp] =', y_free, '+ ( P x  sigmap / ', \
      portfolio.iloc[idx, 1], ') x (', portfolio.iloc[idx, 0],\
      '-', y_free, ')\n\n')

print('\nCapital Allocation Line:\nE[ri] =', y_free,\
     '+ sigmai / ', portfolio.iloc[idx, 1], ' x (', portfolio.iloc[idx, 0],\
      '-', y_free, ')\n\n')

sharpe_max = (portfolio.iloc[idx, 0] - y_free)/portfolio.iloc[idx, 1]
print('Sharpe massimo: ', sharpe_max)


#################### CAPM #########################################

# Trovo rendimenti e analisi del portafogli di mercato
m_expret = portfolio.iloc[idx, 0]
m_vol = portfolio.iloc[idx, 1]**2
market_portfolio_weights = np.array(market_portfolio_weights)
m_ret = np.dot(data_ret, market_portfolio_weights)
m_ret[0] = 0

# Trovo i beta e exp returns dei titoli
beta = []
exp_ret = []

for i in range(30000):
    
    weights = np.random.random(len(tlist) +1 )
    equity_ret = np.dot(data_ret, weights)
    dataframe =  pd.DataFrame({'Equity': equity_ret, 'Market Pfolio': m_ret})
    covariance = dataframe.cov().iloc[0,1] * 250
    
    beta.append(covariance / m_vol)
    exp_ret.append(np.dot(data_ret.mean()*250, weights))
    
capm = pd.DataFrame({'Exp Return': exp_ret, 'Beta': beta})

# Security Market Line
coeff = (m_expret-y_free) / (1 - 0)
x1, y1 = [0, 4.5], [y_free, (coeff + y_free) * 4.5]
x2, y2 = [0, 4.5], [y_free, (coeff + y_free) * 4.5]
sml = str(y_free) + ' + Sigma (p) x ' + str((portfolio.iloc[idx,0]-y_free)\
          / m_expret)

# Plot 
capm.plot(x = 'Beta', y = 'Exp Return', kind = 'scatter', figsize = (15, 6))
plt.plot(x1, y1, x2, y2, marker='o')
plt.text(0.3, 0.3, 'Security Market Line\n' + sml, fontsize = 14)
plt.show()

# Best fit
from tendenza import best_fit
a, b = best_fit(beta, exp_ret)

yfit = []
for i in beta:
    yfit.append(a + b*i)
    
plt.plot(beta, exp_ret, marker = ',', linestyle = '')
plt.plot(beta, yfit)
plt.axis([0, 2, 0, 0.4])
plt.show()

# Tasso risk free estrapolato dalla retta di best fit: a

############### PFOLIO DI MINIMA VARIANZA #############################

min_var = 1
count = 0
for i in pfolio_vol:
    if min_var < i:
        pass
    else:
        min_var = i

pfolio_vol = list(pfolio_vol)
min_var_idx = pfolio_vol.index(min_var)

print('Minima Varianza:\n', portfolio.iloc[min_var_idx])

text = 'Without art-\nMinimum variance portfolio: ' + \
       str(portfolio.iloc[min_var_idx, 1]) + '\nMaximum Sharpe ratio: ' + \
       str(sharpe_max)

fh.write(text)
fh.close()

########################## Portafogli ###################################

counter= 0
idx_01 = []
exp_01 = []
for i in portfolio['Volatility']:
    if round(float(i), 3) == 0.1:
        idx_01.append(counter)
    counter += 1

for i in idx_01:
    exp_01.append(portfolio.iloc[i, 0])

max_exp_01 = list(portfolio['Exp Return']).index(max(list(exp_01)))
print('\n', portfolio.iloc[max_exp_01])

print('\n', max(list(exp_01)))


