# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:12:38 2021

@author: SmirkingJay
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:01:37 2021

@author: SmirkingJay
"""

import gurobipy as gp
import csv
import pandas as pd #for csv reading
import numpy as np 
import matplotlib.pyplot as plt #for plotting

"""
Import your input data for the model
"""
range_t=range(288)
#index_col='Start date/time'
###Import your input data for the model
data_summer=pd.read_csv('AssB_Input_Group1_summer.csv',index_col='Start date/time', parse_dates = True)
data_winter=pd.read_csv('AssB_Input_Group1_winter.csv', index_col='Start date/time', parse_dates = True)
from datetime import datetime as dt


# dynamic electricity prices vector
elec_pric_summer=data_summer['Electricity price [euro/kWh]']
elec_price_winter=data_winter['Electricity price [euro/kWh]']
#household's 15-min PV generation vector
pv_summer=data_summer['PV generation [kW]']
pv_winter=data_winter['PV generation [kW]']
#household's 15-min demand vector
load_summer=data_summer['Residential load [kW]']
load_winter=data_winter['Residential load [kW]']

######## Time-step
Delta_t = 0.25 # 15 minute (0.25 hour) intervals
T=288#number of time-slots (in three days)
range_t=np.array(range(T))
######## Limits on grid and max, min, and initial SOC
Pgridmax = 3 #[kW]
Pbatmax = 4 #[kW]

SoC_min = 0.2 #[-] (battery min state of charge)
SoC_max = 1 #[-] (battery max state of charge)
SoC0 = 0.5 #[-] (initial battery state of charge at the beginning of the day)

C_bat = 13.5 #battery capacity parameter for a Tesla Powerwall rated at 13,5 [kWh]
eff_dis = 0.94 #battery discharging efficeicny
eff_ch = 0.94 #battery charging efficiency

######## Plot power demand and PV generation data
f1 = plt.figure(1)


### Step 1: Create a model
m = gp.Model('Question 2 Summer')

### Step 2: Define variables
######## Define your decision variables for the time horizon using addVars
p_grid = m.addVars(T, lb = -3, ub = 3, name= 'p_grid', vtype=gp.GRB.CONTINUOUS) #-Pgrid,max ≤ Pgrid,t ≤ Pgrid,max = grid constraint
p_batt =m.addVars(T, lb = -4, ub = 4, name= 'p_batt',vtype=gp.GRB.CONTINUOUS) #
e_batt = m.addVars(T, lb = SoC_min*C_bat, ub = SoC_max*C_bat, name= 'e_batt', vtype = gp.GRB.CONTINUOUS) #0 ≤ Ebatt,t  ≤ Bcap 
    
p_ch = m.addVars(T, lb = 0, ub = 4, name = 'p_ch') #0 ≤ Pch,t ≤Pchmax,t
p_disch = m.addVars(T, lb = 0, ub =4 , name = 'p_disch')
    
###Step 3: Add constraints

######## Nonnegative variables 
 
######## Power balance formula
Balance_constraint = m.addConstrs((load_summer[i]==pv_summer[i]+p_grid[i]+p_batt[i] for i in range(T)))

######## Power boundariesrange()
Battery_constraint = m.addConstrs(p_batt[i] == -(p_ch[i]) + (p_disch[i]) for i in range(T))
initial_const=m.addConstr(e_batt[0]==SoC0*C_bat- (-(p_ch[0])*eff_ch + (p_disch[0])/eff_dis) * Delta_t)
Energycont_constrain = m.addConstrs(e_batt[i] == e_batt[i-1] - (-(p_ch[i])*eff_ch + (p_disch[i])/eff_dis) * Delta_t for i in range(1,T) )
#emissions_cons=    m.addConstr((gp.quicksum(data_summer['Marginal emission factor [kg CO2eq/kWh]'][i]*p_grid[i]*Delta_t for i in range(T))) <=15.2)
"""
Step 4: Set objective function
"""

m.setObjective(gp.quicksum(data_summer['Marginal emission factor [kg CO2eq/kWh]'][i]*p_grid[i]*Delta_t for i in range(T)), gp.GRB.MINIMIZE)


"""
Step 5: Solve model
"""

m.optimize()

"""
Step 6: Print variables values for optimal solution
""" 
######## Get the values of the decision variables

m.printAttr('X')
variables=m.getAttr('X')
POWER_GRID_Summer=pd.DataFrame(index=elec_pric_summer.index)
POWER_GRID_Summer['Power']=variables[0:T]
POWER_BATT_Summer=pd.DataFrame(index=elec_pric_summer.index)
POWER_BATT_Summer['Power']=variables[T:2*T]
ENERGY_BATT_Summer=pd.DataFrame(index=elec_pric_summer.index)
ENERGY_BATT_Summer['Energy']=variables[2*T:3*T]
P_CH_Summer=pd.DataFrame(index=elec_pric_summer.index)
P_CH_Summer['Power']=variables[3*T:4*T]
P_DCH_Summer=pd.DataFrame(index=elec_pric_summer.index)
P_DCH_Summer['Power']=variables[4*T:5*T]


#ENERGY_BATT.plot()
#P_CH.plot(ax=ax)
#ax=POWER_BATT.plot()
#POWER_GRID.plot()

POWER_GRID_Summer.index=elec_pric_summer.index
POWER_GRID_Summer.columns=['power']
cost_summer=pd.DataFrame(index=elec_pric_summer.index)
cost_summer['cost']=POWER_GRID_Summer['power']*elec_pric_summer*Delta_t
emissions_summer=pd.DataFrame(index=elec_pric_summer.index)
emissions_summer['emissions']=POWER_GRID_Summer['power']*data_summer['Marginal emission factor [kg CO2eq/kWh]']*Delta_t
total_emissions_summer=emissions_summer.sum()
total_costs_summer=cost_summer.sum()

results_summer = pd.DataFrame( index = range(T))
results_summer['Power Grid']=POWER_GRID_Summer
results_summer['Power Battery']=POWER_BATT_Summer
results_summer['Power Charging']=P_CH_Summer
results_summer['Power Discharging']=P_DCH_Summer
results_summer['Battery Energy']=ENERGY_BATT_Summer

import matplotlib.dates as mdates
fig, ax = plt.subplots(2,1)
fig.set_size_inches(18.5, 10.5)

#fmt_day = mdates.DayLocator()
#fmt_hour = mdates.HourLocator(byhour=[4, 8, 12, 16, 20])

ax[0].plot(results_summer.index, elec_pric_summer, 'b-')
#ax[0].xaxis.set_major_locator(fmt_day)
#ax[0].xaxis.set_minor_locator(fmt_hour)
ax[0].set_title('Electricity price', fontsize = 24)
ax[0].set_ylim(0,0.08)
ax[0].set_ylabel('€/kWh', fontsize = 24)
ax[0].grid(axis='y', linestyle='--')
ax[0].grid(axis='x', which='minor', linestyle='--')
ax[0].grid(axis='x', which='major', linewidth=1.5)

ax[1].plot(results_summer.index, ENERGY_BATT_Summer['Energy'], 'g')
#ax[1].xaxis.set_major_locator(fmt_day)
#ax[1].xaxis.set_minor_locator(fmt_hour)
ax[1].set_title('Battery Energy', fontsize = 24)
#ax[1].set_ylim(0,105)
ax[1].set_ylabel('kWh', fontsize = 24)
ax[1].grid(axis='y', linestyle='--')
ax[1].grid(axis='x', which='minor', linestyle='--')
ax[1].grid(axis='x', which='major', linewidth=1.5)

fig.suptitle("Summer", fontsize = 30)
fig.subplots_adjust(top = 0.92)

fig.tight_layout()
plt.show()
 

fig, ax = plt.subplots(2,1)
fig.set_size_inches(18.5, 10.5)

ax[0].plot(results_summer.index, load_summer, 'k', label='Residential demand')
ax[0].plot(results_summer.index, pv_summer, 'r-', label='Power from PV')
#ax[0].xaxis.set_major_locator(fmt_day)
#ax[0].xaxis.set_minor_locator(fmt_hour)
ax[0].set_title('Residential demand and power from PV system', fontsize = 24)
ax[0].set_ylabel('kW', fontsize = 24)
ax[0].legend(fontsize = 20)
ax[0].grid(axis='y', linestyle='--')
ax[0].grid(axis='x', which='minor', linestyle='--')
ax[0].grid(axis='x', which='major', linewidth=1.5)


ax[1].plot(results_summer.index, POWER_GRID_Summer, 'y', label = 'Grid')
ax[1].plot(results_summer.index, POWER_BATT_Summer, label = 'Battery')
#ax[1].xaxis.set_major_locator(fmt_day)
#ax[1].xaxis.set_minor_locator(fmt_hour)
ax[1].set_title('Power to / from grid and battery', fontsize = 24)
ax[1].set_ylabel('kW', fontsize = 24)
ax[1].set_ylim(-4, 4)
ax[1].legend(fontsize = 20)
ax[1].grid(axis='y', linestyle='--')
ax[1].grid(axis='x', which='minor', linestyle='--')
ax[1].grid(axis='x', which='major', linewidth=1.5)

fig.tight_layout()
fig.subplots_adjust(top = 0.9)


###WINTER

w = gp.Model('Question 2 Winter')

p_grid = w.addVars(T, lb = -3, ub = 3, name= 'p_grid', vtype=gp.GRB.CONTINUOUS) #-Pgrid,max ≤ Pgrid,t ≤ Pgrid,max = grid constraint
p_batt = w.addVars(T, lb = -4, ub = 4, name= 'p_batt',vtype=gp.GRB.CONTINUOUS) #
e_batt = w.addVars(T, lb = SoC_min*C_bat, ub = SoC_max*C_bat, name= 'e_batt', vtype = gp.GRB.CONTINUOUS) #0 ≤ Ebatt,t  ≤ Bcap 
    
p_ch = w.addVars(T, lb = 0, ub = 4, name = 'p_ch') #0 ≤ Pch,t ≤Pchmax,t
p_disch = w.addVars(T, lb = 0, ub = 4, name = 'p_disch')

Balance_constraint = w.addConstrs((load_winter[i]==pv_winter[i]+p_grid[i]+p_batt[i] for i in range(T)))

######## Power boundariesrange()
Battery_constraint = w.addConstrs(p_batt[i] == -(p_ch[i]) + (p_disch[i]) for i in range(T))
initial_const=w.addConstr(e_batt[0]==SoC0*C_bat- (-(p_ch[0])*eff_ch + (p_disch[0])/eff_dis) * Delta_t)
Energycont_constrain = w.addConstrs(e_batt[i] == e_batt[i-1] - (-(p_ch[i])*eff_ch + (p_disch[i])/eff_dis) * Delta_t for i in range(1,T) )



w.setObjective(gp.quicksum(data_winter['Marginal emission factor [kg CO2eq/kWh]'][i]*p_grid[i]*Delta_t for i in range(T)), gp.GRB.MINIMIZE)


"""
Step 5: Solve model
"""

w.optimize()

"""
Step 6: Print variables values for optimal solution
""" 
######## Get the values of the decision variables

w.printAttr('X')
variables=w.getAttr('X')
POWER_GRID_Winter=pd.DataFrame(variables[0:T])
POWER_BATT_Winter=pd.DataFrame(variables[T:2*T])
ENERGY_BATT_Winter=pd.DataFrame(variables[2*T:3*T])
P_CH_Winter=pd.DataFrame(variables[3*T:4*T])
P_DCH_Winter=pd.DataFrame(variables[4*T:5*T])

#ENERGY_BATT.plot()
#P_CH.plot(ax=ax)
#ax=POWER_BATT.plot()
#POWER_GRID.plot()

POWER_GRID_Winter.index=elec_price_winter.index
POWER_GRID_Winter.columns=['power']
cost_Winter=pd.DataFrame(index=elec_price_winter.index)
cost_Winter['cost']=POWER_GRID_Winter['power']*elec_price_winter*Delta_t
emissions_winter=pd.DataFrame(index=elec_price_winter.index)
emissions_winter['emissions']=POWER_GRID_Winter['power']*data_winter['Marginal emission factor [kg CO2eq/kWh]']*Delta_t
total_emissions_winter=emissions_winter.sum()
total_costs_winter=cost_Winter.sum()