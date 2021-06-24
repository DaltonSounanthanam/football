#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:19:16 2021

@author: dalton
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt

data = pd.read_csv('play_information.csv')


EndZone = []
FairCatch = []
PuntYard = []
PunterName = []
ReturnerName = []
ReturnYard = []
Penalty = []
Blocked = []
Touchdown = []
Pass = []
Fumbled = []
OutOfBounds = []

for i in range(data.shape[0]):
    tmp = data.iloc[i]
    yardline = tmp['YardLine']
    possteam = tmp['Poss_Team']
    playdescription = tmp['PlayDescription']
    #print(playdescription)
    liste_playdescription = playdescription.split(' ')
    
    if 'BLOCKED' in liste_playdescription or 'blocked' in liste_playdescription or 'SAFETY' in liste_playdescription or 'Blocked' in liste_playdescription or 'block' in liste_playdescription:
        Blocked.append(1)
    else:
        Blocked.append(0)
        
    if 'TOUCHDOWN' in liste_playdescription:
        Touchdown.append(1)
    else:
        Touchdown.append(0)
        
    if 'pass' in liste_playdescription or 'Fake' in liste_playdescription:
        Pass.append(1)
    else:
        Pass.append(0)
        
    if 'FUMBLES' in liste_playdescription or 'Fumbled' in liste_playdescription:
        Fumbled.append(1)
    else:
        Fumbled.append(0)
        
        
    if 'punts' in liste_playdescription:
        start = liste_playdescription.index('punts')
        PuntYard.append(int(liste_playdescription[start + 1]))
        PunterName.append(liste_playdescription[start - 1])
        liste_playdescription[start + 2] = 'foo'
    else:
        PuntYard.append(0)
        PunterName.append('/NA')
    
    returnyard = 0
    returnname = '/NA' 
    
    if 'PENALTY' in liste_playdescription or 'Penalty' in liste_playdescription or 'declined' in liste_playdescription:
        Penalty.append(1)
        returnyard = 0
        returnname = '/NA'
    else:
        Penalty.append(0)
        
    if playdescription.find('end zone') >= 0:
        EndZone.append(1)
        returnyard = 0
        returnname = '/NA'
    else:
        EndZone.append(0)
        
    if playdescription.find('fair catch') >= 0:
        FairCatch.append(1)
        returnyard = 0
        returnname = '/NA'
    else:
        FairCatch.append(0)
    
    
    if playdescription.find('out of bounds') >= 0:
        OutOfBounds.append(1)
    else:
        OutOfBounds.append(0)
    
    if 'yards' in liste_playdescription:    
        index = liste_playdescription.index('yards')
        returnyard = int(liste_playdescription[index - 1])
        returnname = liste_playdescription[index - 6]
        
    ReturnYard.append(returnyard)
    ReturnerName.append(returnname)
    
data['Punt_Yard'] = PuntYard
data['Punter'] = PunterName
data['Fair_Catch'] = FairCatch
data['End_Zone'] = EndZone
data['Penalty'] = Penalty
data['Blocked'] = Blocked
data['Return_Yard'] = ReturnYard
data['Pass'] = Pass
data['Fumble'] = Fumbled
data['Out_Of_Bounds'] = OutOfBounds

#data.to_csv('play_information_ext.csv')




### Punt distribution across the field or Where do teams punt the most ? or Where NFL offenses are stopped by NFL defenses ? ###

punt_distance = []


for i in range(data.shape[0]):
    tmp = data.iloc[i]
    yardline = tmp['YardLine'].split(' ')
    possteam = tmp['Poss_Team']
    if possteam in yardline:
        punt_distance.append(int(yardline[1]))
    else:
        punt_distance.append(100 - int(yardline[1]))
        
punt_distance_avg = np.mean(punt_distance)

punt_distance_std = np.std(punt_distance)
        
X = np.asarray(punt_distance).reshape(-1,1)
kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.5).fit(X)

yards = np.linspace(start = 0, stop = 100, num = 300).reshape(-1,1)
probs = kde.score_samples(yards)

fig1, ax1 = plt.subplots()
ax1.plot(yards, np.exp(probs), ls = '-')
ax1.set_title('Punt distance distribution')
ax1.set_xlabel('Yards')
ax1.set_ylabel('Probabilty') 
        
### Punt distribution across NFL teams or Which NFL team has the worst offense + (return team) ? or Which NFL team has the best defense + (punting, kicking team) or Super aggressive teams that go on 4th down ###

team_list = set(data['Poss_Team'].tolist())
counter = dict.fromkeys(team_list, 0)

for i in range(data.shape[0]):
    tmp = data.iloc[i]
    possteam = tmp['Poss_Team']
    counter[possteam] = counter[possteam]+ 1

fig2, ax2 = plt.subplots()
ax2.bar(counter.keys(), counter.values())
ax2.set_title('Number of punts for each teams' )




### What does it take to get a fair catch ? ###

data['Punt_Dist'] = punt_distance
data['Punt_Dist'] = 100 - data['Punt_Dist']

fair_catches = data[data['Fair_Catch'] == 1]
end_zones = data[(data['End_Zone'] == 1) ]
returns = data[ ((data['Fair_Catch'] == 0) & (data['End_Zone'] == 0)  & (data['Penalty'] == 0) & (data['Blocked'] == 0) & (data['Pass'] == 0) & (data['Fumble'] == 0) & (data['Out_Of_Bounds'] == 0) & (data['Punt_Yard'] > 0) ) ] 
outofbounds = data[ ((data['Fair_Catch'] == 0) & (data['End_Zone'] == 0)  & (data['Penalty'] == 0) & (data['Blocked'] == 0) & (data['Pass'] == 0) & (data['Fumble'] == 0) & (data['Out_Of_Bounds'] == 1) ) ]


'''
fair_catches_dist = []
end_zones_dist = []
returns_dist = []
outofbounds_dist = []

for i in range(fair_catches.shape[0]):
    tmp = fair_catches.iloc[i]
    yardline = tmp['YardLine'].split(' ')
    possteam = tmp['Poss_Team']
    if possteam in yardline:
        fair_catches_dist.append(100 - int(yardline[1]))
    else:
        fair_catches_dist.append( int(yardline[1]))

for i in range(end_zones.shape[0]):
    tmp = end_zones.iloc[i]
    yardline = tmp['YardLine'].split(' ')
    possteam = tmp['Poss_Team']
    if possteam in yardline:
        end_zones_dist.append(100 - int(yardline[1]))
    else:
        end_zones_dist.append( int(yardline[1]))    
        
for i in range(returns.shape[0]):
    tmp = returns.iloc[i]
    yardline = tmp['YardLine'].split(' ')
    possteam = tmp['Poss_Team']
    if possteam in yardline:
        returns_dist.append(100 - int(yardline[1]))
    else:
        returns_dist.append( int(yardline[1]))   
    
for i in range(outofbounds.shape[0]):
    tmp = outofbounds.iloc[i]
    yardline = tmp['YardLine'].split(' ')
    possteam = tmp['Poss_Team']
    if possteam in yardline:
        outofbounds_dist.append(100 - int(yardline[1]))
    else:
        outofbounds_dist.append( int(yardline[1]))

'''        
fig3, ax3 = plt.subplots()
ax3.scatter(fair_catches['Punt_Dist'], fair_catches['Punt_Yard'].tolist(), c = 'green', marker = '.' )
ax3.scatter(end_zones['Punt_Dist'], end_zones['Punt_Yard'].tolist(), c = 'red', marker = '.' )
ax3.scatter(returns['Punt_Dist'], returns['Punt_Yard'].tolist(), c = 'orange', marker = '.')
ax3.scatter(outofbounds['Punt_Dist'], outofbounds['Punt_Yard'].tolist(), c = 'c', marker = '.')
ax3.set_ylabel('punt yards')
ax3.set_xlabel('yards from goal line')
ax3.set_title('')


### When can you get at least 10 yards (win a 1st down for your offense) out of a punt return ?


# what happens when you punt from your own redzone ?
returns_rz = returns[returns['Punt_Dist'] > 60]
returns_positives = returns_rz[returns_rz['Return_Yard'] >= 10 ]
returns_undetermined = returns_rz[(returns_rz['Return_Yard'] < 10) & (returns_rz['Return_Yard'] > 2) ]
returns_negatives = returns_rz[returns_rz['Return_Yard'] <= 2]
returns_zeros = returns_rz[returns_rz['Return_Yard'] == 0]

fig4, ax4 = plt.subplots()
ax4.scatter(returns_positives['Punt_Dist'], returns_positives['Punt_Yard'].tolist(), c = 'green', marker = '.')
ax4.scatter(returns_negatives['Punt_Dist'], returns_negatives['Punt_Yard'].tolist(), c = 'red', marker = '.')
#ax4.scatter(returns_undetermined['Punt_Dist'], returns_undetermined['Punt_Yard'].tolist(), c = 'blue', marker = '.')
#ax4.scatter(returns_zeros['Punt_Dist'], returns_zeros['Punt_Yard'].tolist(), c = 'orange', marker = '.')
ax4.set_xlabel('distance from goal line')
ax4.set_ylabel('punt yards')
ax4.set_title('Positive vs Negative returns')


# What happens when you punt far from your redzone ?

returns_nrz = returns[returns['Punt_Dist'] <= 60]
returns_positives = returns_nrz[returns_nrz['Return_Yard'] >= 10 ]
returns_undetermined = returns_nrz[(returns_nrz['Return_Yard'] < 10) & (returns_nrz['Return_Yard'] > 2) ]
returns_negatives = returns_nrz[returns_nrz['Return_Yard'] <= 2]
returns_zeros = returns_nrz[returns_nrz['Return_Yard'] == 0]

fig5, ax5 = plt.subplots()
ax5.scatter(returns_positives['Punt_Dist'], returns_positives['Punt_Yard'].tolist(), c = 'green', marker = '.')
ax5.scatter(returns_negatives['Punt_Dist'], returns_negatives['Punt_Yard'].tolist(), c = 'red', marker = '.')
#ax5.scatter(returns_undetermined['Punt_Dist'], returns_undetermined['Punt_Yard'].tolist(), c = 'blue', marker = '.')
#ax5.scatter(returns_zeros['Punt_Dist'], returns_zeros['Punt_Yard'].tolist(), c = 'orange', marker = '.')
ax5.set_xlabel('distance from goal line')
ax5.set_ylabel('punt yards')
ax5.set_title('Positive vs Negative returns')

### study football malpractice, blocked punt and penalties ###

### Fair catch vs Returning punt, when we must not return punt ? ###


### Analyse punting and returning for each team ###

#data[data['Poss_Team'] == '']

### Quest to the perfect punt, an analytical perspective ###

