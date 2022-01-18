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

###Import your input data for the model
data_summer=pd.read_csv('AssB_Input_Group1_summer.csv', index_col='Start date/time', sep = ',', parse_dates = True)
data_winter=pd.read_csv('AssB_Input_Group1_winter.csv', index_col='Start date/time', sep = ',', parse_dates = True)

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
emissions_cons= m.addConstr((gp.quicksum(data_summer['Marginal emission factor [kg CO2eq/kWh]'][i]*p_grid[i]*Delta_t for i in range(T))) <=-9.17)
"""
Step 4: Set objective function
"""

m.setObjective(gp.quicksum(elec_pric_summer[i]*p_grid[i]*Delta_t for i in range(T)), gp.GRB.MINIMIZE)


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
POWER_GRID_Summer=pd.DataFrame(variables[0:T])
POWER_BATT_Summer=pd.DataFrame(variables[T:2*T])
ENERGY_BATT_Summer=pd.DataFrame(variables[2*T:3*T])
P_CH_Summer=pd.DataFrame(variables[3*T:4*T])
P_DCH_Summer=pd.DataFrame(variables[4*T:5*T])

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
emissions_cons=    w.addConstr((gp.quicksum(data_winter['Marginal emission factor [kg CO2eq/kWh]'][i]*p_grid[i]*Delta_t for i in range(T)))<=2.6)


w.setObjective(gp.quicksum(elec_price_winter[i]*p_grid[i]*Delta_t for i in range(T)), gp.GRB.MINIMIZE)


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

summer_pareto=pd.read_csv('Summer_pareto.csv' )

winter_pareto=pd.read_csv('Winter_pareto.csv' )


  
###plots
summer_pareto.plot(x='costs', y='emissions', marker='.',legend=None, markersize=10, kind='line')
plt.xlabel('Total costs [€]') 
plt.ylabel('Total marginal emissions [kg CO2]') 
plt.title('Pareto frontier for summer days')
#plt.xlim(-1.3,-0.3)
#plt.ylim(-16,8)
plt.show()

winter_pareto.plot(x='costs', y='emissions', legend=None, marker='.', markersize=10, kind='line')
plt.xlabel('Total costs [€]') 
plt.ylabel('Total marginal emissions [kg CO2]') 
plt.title('Pareto frontier for winter days')
#plt.xlim(0.1,1.1)
#plt.ylim(-3,20)
plt.show()