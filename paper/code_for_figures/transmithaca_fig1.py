import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import ode
import math 
import pandas as pd
import copy
from definitions_visit_model import *
import random
import operator


'''
TRANSMITHACA: Create ensemble using an SIR model.
'''

# Plot thought-up example
compartments_SIR = "SIR"

# Make array of cities
cities =  ['Whole country']
N_cities = [1000000.]
N_dict = {cities[0] :  N_cities[0]}


# Define parameters :
infection_rate_SIR = 1.1

recovery_rate_SIR = 1


parameters_SIR = {city: {'infection_rate':infection_rate_SIR,'recovery_rate':recovery_rate_SIR} for city in N_dict.keys()}

# Define initial conditions
# Load mobility data

Mobility_basic = np.zeros((len(N_dict.keys()),len(N_dict.keys())))

dic_solution_SIR = {}

for exp in np.arange(0,150,3) :
    
    I_init_SIR = 100#(exp+1)*200

    initial_conditions_SIR = {city: {'S':N_dict[city]-I_init_SIR,'I':I_init_SIR,'R':0} for city in N_dict.keys()}

    # Create dictionary for solved ODEs
    solution_SIR = create_solutions_dic(initial_conditions_SIR)

    # Make initial conditions..
    y0_SIR = give_initial_conditions(model=compartments_SIR,cities=cities,initial_conditions=initial_conditions_SIR)
    y0_SIR_start = copy.deepcopy(y0_SIR)
    
    # Give time interval
    t_fin_SIR = np.array([])#np.arange(0,250,0.1)

    # Define linear rates..
    R_linear_SIR = make_linear_rate_matrix(model=compartments_SIR,cities=cities,parameters=parameters_SIR)

    Mobility = Mobility_basic#_dic[tend-1]


    # Make model and solve 
    model = Commuter_model()

    tmax_SIR = 3*110
    for tend in np.arange(1,tmax_SIR,1) :
        # Choose correct mobility

        t = np.arange(tend-1,tend,0.1)
        if (tend >= exp+1) :
            if ( tend == exp+1) :
                y0_SIR = copy.deepcopy(y0_SIR_start)
            newly_solved_SIR = model.Solve_commuter_model(t,y0_SIR,R_linear_SIR, compartments_SIR, cities,Mobility,N_cities, parameters_SIR )
        else : 
            
            newly_solved_siR = {'S': np.ones(len(t))*N_dict['Whole country'],'I':np.zeros(len(t)),'R':np.zeros(len(t))}
        # From the solved ODEs, concatenate to total solutions and get new initial conditions
        solution_SIR,y0_SIR = concatenate_to_solutions_dic(model=compartments_SIR,cities=cities,solutions=solution_SIR,newly_solved=newly_solved_SIR)

        # Save time for plots.
        t_fin_SIR = np.concatenate((t_fin_SIR,t))

    dic_solution_SIR[exp] = copy.deepcopy(solution_SIR['Whole country']['I'])

'''
Calculate fixed-time statistics
'''

# Dictionary for FT percentiles. Compute 25th, 50th, 75th.
percentiles_SIR = {}
percentiles_SIR[25] = []
percentiles_SIR[75] = []
percentiles_SIR[50] = []

# Also save time steps.
percentile_time_SIR = []

for time in range (tend*10 ) : 
    # Save time
    percentile_time_SIR.append(time)
    
    # Array to keep track of all values on the time step
    arr_time = []
    for sample in list(dic_solution_SIR.keys())[::2] :
        arr_time.append(dic_solution_SIR[sample][time])
    # Sort values wrt size
    arr_time.sort()

    # Find value constituting the pth percentile
    for p in percentiles_SIR.keys() :
        percentiles_SIR[p].append(arr_time[int(len(arr_time)*p/100)])

t_plot = np.arange(0,tend,0.1)



'''
Fig. 1B: QUICK GROWTH, SLOW DECAY
'''

dic_QGSD = {}
N_QGSD = int(len(dic_solution_SIR)/2)
plt.figure(figsize=(3.36*2,3.36/1.5))
decay_const = 0.002
for curve in range (N_QGSD) :
    hit = (curve*6+20)*10
    
    dic_QGSD[curve] = np.zeros(len(t_plot))
    for t in range (hit,len(t_plot)) :
        dic_QGSD[curve][t] += math.exp(-decay_const*(t-hit))
    
    plt.plot(t_plot,dic_QGSD[curve],'C0',alpha=0.2)
    
percentiles_QGSD = {}
percentiles_QGSD[25] = []
percentiles_QGSD[75] = []
percentiles_QGSD[50] = []


percentile_time_QGSD = []
for time in range (tend*10 ) : 
    percentile_time_QGSD.append(time)
    arr_time = []
    for sample in dic_QGSD.keys() :
        arr_time.append(dic_QGSD[sample][time])
    arr_time.sort()
    for p in percentiles_QGSD.keys() :
        percentiles_QGSD[p].append(arr_time[int(len(arr_time)*p/100)])  

   
'''
PLOT RESULTS
'''

# Settings
EVEN_SMALLER_SIZE = 5.5
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=EVEN_SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Linewith
FT_linewidth = 1


# convert cm to inches
def cm_to_inch(cm) :
    return 0.3937007874*cm

# Get placement of inset labels right.
def h_text(xlim) :
    return(xlim[0] + (xlim[1]-xlim[0])/40)
def v_text(ylim) :
    return(ylim[1] - (ylim[1]-ylim[0])/10)

# Transmithaca
fig_figsize = (cm_to_inch(9),cm_to_inch(6))

figs,axs=plt.subplots(nrows=2,ncols=1,figsize=fig_figsize,sharey=False,sharex=True)

# Plot curves
for exp in list(dic_solution_SIR.keys())[::2] :
    axs[0].plot(t_plot,dic_solution_SIR[exp],'C0',alpha=0.2)

# Plot percentiles
axs[0].fill_between(t_plot,percentiles_SIR[25],percentiles_SIR[75],color='grey',alpha=0.2,label='25%-75%')

# Plot median
axs[0].plot(t_plot,percentiles_SIR[50],'k',linewidth=FT_linewidth,label='median')

# ticks and labels    
axs[0].set_xlim([0,280])
axs[0].set_ylabel('Infectious')

# Inset label
htext = h_text(axs[0].get_xlim())
vtext0 = v_text(axs[0].get_ylim())
axs[0].text(htext,vtext0,'A',horizontalalignment='left',verticalalignment='top',fontsize=14)




# B: QGSD
for curve in range (N_QGSD) :
    axs[1].plot(t_plot,dic_QGSD[curve],'C0',alpha=0.2)
    

axs[1].fill_between(t_plot,percentiles_QGSD[25],percentiles_QGSD[75],color='grey',alpha=0.2,label='25%-75%')

axs[1].plot(t_plot,percentiles_QGSD[50],'k',linewidth=FT_linewidth,label='median')    
    
    
# Ticks and lebales
vtext1 = v_text(axs[1].get_ylim())

axs[1].text(htext,vtext1,'B',horizontalalignment='left',verticalalignment='top',fontsize=14)

axs[1].set_ylabel('Load [a.u.]')
axs[1].set_xlabel('Day')
axs[1].legend(loc=1,frameon=False)

figs.align_ylabels(axs[:])

plt.tight_layout()
plt.savefig('Figures/Fig1.png',dpi=400)
