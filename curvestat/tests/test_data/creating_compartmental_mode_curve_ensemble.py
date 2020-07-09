import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import ode
import math 
import copy
from definitions_visit_model import *
import random

'''
First create 'data' using a simple compartmental model.

Danish data was created using another model, see
- description at https://files.ssi.dk/teknisk-gennemgang-af-modellerne-10062020
- code at: https://github.com/laecdtu/C19DK
'''
# -----
# CHOOSING INITIAL CONDITIONS AND PARAMETERS
# -----

# Define compartments 
compartments = "SEIHR"

# Make array of cities
cities =  ['Whole country']


# Population
N_cities = [1000000.]
N_dict = {cities[0] :  N_cities[0]}


# Define parameters [see 'definitions_visit_model' for SEIHR model description]:
infection_rate = (1/7)*2.5
infectious_rate =1/7

base_rate = 1/7
hospitalization_fraction = 0.2
recovery_rate = base_rate*(1-hospitalization_fraction)
hospitalization_rate = base_rate*hospitalization_fraction
discharge_rate = 1/21.

infection_rate_original = infection_rate + 0



parameters = {city: {'infection_rate':infection_rate,'infectious_rate':infectious_rate,'recovery_rate':recovery_rate,
                     'hospitalization_rate':hospitalization_rate,'discharge_rate':discharge_rate} for city in N_dict.keys()}

# Define initial conditions [at time 0, how many Exposed and Infectious are present in population?]
E_init = 60
I_init = 15
initial_conditions = {city: {'S':N_dict[city]-E_init-I_init,'E':E_init,'I':I_init,'H':0,'R':0} for city in N_dict.keys()}

# Load mobility data
Mobility_basic = np.zeros((len(N_dict.keys()),len(N_dict.keys())))


# -----
# INTEGRATING COMPARTMENTAL MODEL
# -----

# Create dictionary for solved ODEs
solution = create_solutions_dic(initial_conditions)

# Make initial conditions..
y0 = give_initial_conditions(model=compartments,cities=cities,initial_conditions=initial_conditions)

# Give time interval
t_fin = np.array([])#np.arange(0,250,0.1)

# Define linear rates..
R_linear = make_linear_rate_matrix(model=compartments,cities=cities,parameters=parameters)

Mobility = Mobility_basic#_dic[tend-1]


# Make model and solve 
model = Commuter_model()

tmax = 110
for tend in np.arange(1,tmax,1) :
    # Choose correct mobility
    
    t = np.arange(tend-1,tend,0.1)
    newly_solved = model.Solve_commuter_model(t,y0,R_linear, compartments, cities,Mobility,N_cities, parameters )
    
    # From the solved ODEs, concatenate to total solutions and get new initial conditions
    solution,y0 = concatenate_to_solutions_dic(model=compartments,cities=cities,solutions=solution,newly_solved=newly_solved)
    
    # Save time for plots.
    t_fin = np.concatenate((t_fin,t))
    
    if (tend == 60) :
        
        infection_rate = infection_rate/5
        parameters = {city: {'infection_rate':infection_rate,'infectious_rate':infectious_rate,'recovery_rate':recovery_rate,
                             'hospitalization_rate':hospitalization_rate,'discharge_rate':discharge_rate} for city in N_dict.keys()}

# Save result
save_data = {'data-H':solution[cities[0]]['H'][0::10]}
np.save('datapoints_compartmental_simulation.npy', save_data) 





'''
Simulate future: Creating the curve ensemble
'''

# -----
# First
# -----

# Number of samples required
N_samples = 2

# How accurate must a curve predict final day's hospitalized cases to be saved to the ensemble
H_diff_allowed = 50

# Variable to count whether we have saved N_samples curves to the ensemble
samples_saved = 0

# Dictionary of saved parameters and initial conditions.
dic_saved = {}

# Dictionary for curves
dic_solutions = {}

# Uncertainty on all parameters (except hospitalization probability)
error = 0.25
while samples_saved < N_samples :


    
    
    # Define parameters :
    infectious_rate =1/(7*(1+random.uniform(-1,1)*error))    
    
    base_rate = 1/(7*(1+random.uniform(-1,1)*error))
    
    #large uncertainty on how many cases get hospitalized
    hospitalization_fraction = random.uniform(0.01,0.30) 

    recovery_rate = base_rate*(1-hospitalization_fraction)
    hospitalization_rate = base_rate*hospitalization_fraction    
    
    infection_rate = (hospitalization_rate+recovery_rate)*2.5*(1+random.uniform(-1,1)*error)
    discharge_rate = 1/(21.*(1+random.uniform(-1,1)*error)) # error is on the expected number of days a patient is hospitalized for

    infection_rate_original = infection_rate + 0



    parameters = {city: {'infection_rate':infection_rate,'infectious_rate':infectious_rate,'recovery_rate':recovery_rate,
                         'hospitalization_rate':hospitalization_rate,'discharge_rate':discharge_rate} for city in N_dict.keys()}

    # Define initial conditions
    E_init = 60*(1+random.uniform(-1,1)*error)
    I_init = 15*(1+random.uniform(-1,1)*error)
    initial_conditions = {city: {'S':N_dict[city]-E_init-I_init,'E':E_init,'I':I_init,'H':0,'R':0} for city in N_dict.keys()}

    # Create dictionary for solved ODEs
    solution = create_solutions_dic(initial_conditions)

    # Make initial conditions..
    y0 = give_initial_conditions(model=compartments,cities=cities,initial_conditions=initial_conditions)


    # Define linear rates..
    R_linear = make_linear_rate_matrix(model=compartments,cities=cities,parameters=parameters)

    Mobility = Mobility_basic

    # Make model and solve 
    model = Commuter_model()

    tmax = 110
    for tend in np.arange(1,tmax,1) :

        t = np.arange(tend-1,tend,0.1)
        newly_solved = model.Solve_commuter_model(t,y0,R_linear, compartments, cities,Mobility,N_cities, parameters )

        # From the solved ODEs, concatenate to total solutions and get new initial conditions
        solution,y0 = concatenate_to_solutions_dic(model=compartments,cities=cities,solutions=solution,newly_solved=newly_solved)



        if (tend == 60) :
            infection_rate_original = infection_rate + 0
            infection_rate = infection_rate*(0.2*(1+random.uniform(-1,1)*error))
            parameters = {city: {'infection_rate_original':infection_rate_original,'infection_rate':infection_rate,'infectious_rate':infectious_rate,'recovery_rate':recovery_rate,
                                 'hospitalization_rate':hospitalization_rate,'discharge_rate':discharge_rate} for city in N_dict.keys()}
        
    # Calculate difference in hospitalized cases between simulation and data on last day of data:
    diff = abs(solution['Whole country']['H'][-1]-save_data['data-H'][-1])

    # If difference is small enough, save curve to ensemble
    if (diff < H_diff_allowed) :
        dic_saved[samples_saved] = {'parameters':copy.deepcopy(parameters),'initial_conditions':initial_conditions}
        print(samples_saved,"out of",N_samples,"saved")

        # Save curve 
        for curvevar in solution[cities[0]].keys() :
            solution[cities[0]][curvevar] = solution[cities[0]][curvevar][0::10]
        dic_solutions[samples_saved] = copy.deepcopy(solution)
        samples_saved +=1

np.save('configurations_dictionaries.npy', dic_saved) 
np.save('curves_compartmental_simulations.npy', dic_solutions) 
