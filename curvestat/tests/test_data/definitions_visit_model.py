import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import ode
import math 
import pandas as pd
import copy

def get_compartment_id(compartments,compartment) :
    return compartments.index(compartment)

def get_rates(model,which_processes) :
    
    '''
    SI MODEL
    '''
    if ( model == 'SI' ) :
        linear_processes = [
        ]

        nonlinear_processes = [
            ('S','I','S','-infection_rate'),
            ('S','I','I','+infection_rate'),
        ]

    '''
    SIR MODEL
    '''
    if ( model == 'SIR' ) :
        linear_processes = [
            ('I','I','-recovery_rate'),            
            ('I','R','+recovery_rate'),
        ]

        nonlinear_processes = [
            ('S','I','S','-infection_rate'),
            ('S','I','I','+infection_rate'),
        ]

    '''
    SEIR MODEL
    '''
    if ( model == 'SEIR' ) :
        linear_processes = [
            ('E','E','-infectious_rate'),            
            ('E','I','+infectious_rate'),            

            ('I','I','-recovery_rate'),            
            ('I','R','+recovery_rate'),
        ]

        nonlinear_processes = [
            ('S','I','S','-infection_rate'),
            ('S','I','E','+infection_rate'),
        ]

    '''
    SEIHR MODEL
    '''
    if ( model == 'SEIHR' ) :
        linear_processes = [
            ('E','E','-infectious_rate'),            
            ('E','I','+infectious_rate'),            

            ('I','I','-recovery_rate'),            
            ('I','R','+recovery_rate'),


            ('I','I','-hospitalization_rate'),            
            ('I','H','+hospitalization_rate'),

            ('H','H','-discharge_rate'),            
            ('H','R','+discharge_rate'),


        ]

        nonlinear_processes = [
            ('S','I','S','-infection_rate'),
            ('S','I','E','+infection_rate'),
        ]        

    '''
    SEAIHDR MODEL
    '''

    if ( model == 'SEAIHDR' ) :
        linear_processes = [
            

            ("E", "E", '-asymptomatic_rate'),
            ("E", "A", '+asymptomatic_rate'),

            ("A", "A", '-symptomatic_rate'),
            ("A", "I", '+symptomatic_rate'),

            ("A", "A", '-asymptomatic_recovery_rate'),
            ("A", "R", '+asymptomatic_recovery_rate'),

            ("I", "I", '-symptomatic_recovery_rate'),
            ("I", "R", '+symptomatic_recovery_rate'),
            
            ("I","I",'-hospitalization_rate'),
            ("I","H",'+hospitalization_rate'),
            
            ("H","H",'-fatality_rate'),
            ("H","D",'+fatality_rate'),
            
            ("H","H",'-discharge_rate'),
            ("H","R",'+discharge_rate'),


        ]

        nonlinear_processes = [
            ('S','I','S','-symptomatic_infection_rate'),
            ('S','I','E','+symptomatic_infection_rate'),
            ('S','A','S','-asymptomatic_infection_rate'),
            ('S','A','E','+asymptomatic_infection_rate'),            
        ]


    if (which_processes == 'linear') :
        return linear_processes
    else : 
        return nonlinear_processes



def create_solutions_dic(initial_conditions) :
    # Makes a new dictionary from the initial_conditions dictionary. Replaces values with numpy arrays.
    solution = copy.deepcopy(initial_conditions)
    for city in solution.keys() :
        for compartment in solution[city].keys() :
            solution[city][compartment] = np.array([])
            
    return solution

def concatenate_to_solutions_dic(model,cities, solutions,newly_solved):

    y0 = np.zeros(len(model)*len(cities))
    
    for city in solutions.keys() :
        city_id = get_compartment_id(cities,city)
        for compartment in solutions[city].keys () :
            compartment_id = get_compartment_id(model,compartment)
            
            solutions[city][compartment] = np.concatenate((solutions[city][compartment],
                                                           newly_solved[city_id*len(model)+compartment_id]))
            
            y0[city_id*len(model)+compartment_id] = newly_solved[city_id*len(model)+compartment_id][-1]
            
    
    return solutions,y0 



def make_linear_rate_matrix(model,cities,parameters) :
    '''
    Makes the linear rate matrix. This only needs to be made once if the disease stays the same.
    '''
    
    
    # Make linear rate matrix
    R_linear = np.zeros((len(cities)*len(model),len(cities)*len(model)))
    
    # get linear processes
    linear_processes = get_rates(model=model,which_processes='linear')


    # Insert parameters in the right places
    for city in cities :
        city_offset = get_compartment_id(cities,city)*len(model)
        
        for process in linear_processes :
            # First entry in process is the sign of the rate; rest is parameter name
            sign_rate = float(process[2][0]+'1')
            parameter_name = process[2][1:]
            
            # Insert current rate in the right place with the right sign
            R_linear[city_offset+get_compartment_id(model,process[1]),
                     city_offset+get_compartment_id(model,process[0])] += sign_rate*parameters[city][parameter_name]

    return R_linear

def give_initial_conditions(model,cities,initial_conditions) :
    
    '''
    Takes initial conditions from a dictionary and turn them into an array
    '''
    
    # Define the array
    initial_conditions_array = np.zeros(len(cities)*len(model))
    
    # Loop over cities and get index of city in question
    for city in initial_conditions.keys() :
        city_index = cities.index(city)
        # Loop over different compartments, each of which has a defined initial condition; get compartment id.
        for compartment in initial_conditions[city].keys() :
            compartment_index = list(model).index(compartment)
            
            # Insert initial condition in the right place in array.
            initial_conditions_array[city_index*len(model)+compartment_index] = initial_conditions[city][compartment] +0
    return initial_conditions_array


def make_S_matrix(model,Mobility,y,N) :
    # Delete Mobility matrix diagonal
    np.fill_diagonal(Mobility, 0)
    
    S_vec = y[get_compartment_id(list(model),'S')::len(model)]
    S_matrix = np.transpose(np.transpose(Mobility)*S_vec*np.reciprocal(N))
    S_matrix += np.diag(S_vec-sum(S_matrix))

    return S_matrix

def make_X_eff(model,Mobility,y,N,specific_compartment) :
    # model: compartmental model; string, such as 'SIR'
    # Mobility: Flux matrix
    # y : state array
    # N : array with entry i = Number of mobile people in metapopulation i
    # specific_compartment : compartment X, e.g. 'I'

    # Delete Mobility matrix diagonal
    np.fill_diagonal(Mobility, 0)

    X_vec = y[get_compartment_id(list(model),specific_compartment)::len(model)]

    X_eff = np.zeros(len(X_vec))
    X_eff += X_vec
    X_eff -= np.transpose(Mobility).dot(np.ones(len(X_eff)))*(X_vec*np.reciprocal(N))
    X_eff += Mobility.dot(X_vec*np.reciprocal(N))

    return X_eff

def get_nonlinear_terms(model,cities,Mobility,y,N,parameters) :

    # define result array
    nonlinear_terms = np.zeros((len(y)))

    # get nonlinear processes
    nonlinear_processes = get_rates(model=model,which_processes='nonlinear')
        
    # make S_matrix (keep track of number of S from metapop i that are in j this day)
    S_matrix = make_S_matrix(model,Mobility,y,N)

    # make X_eff
    X_eff_dic = {}
    for process in nonlinear_processes :
        infectious_compartment = process[1]
        sign_rate = float(process[3][0]+'1')
        parameter_name = process[3][1:]

        parameter_arr = np.zeros(len(list(parameters.keys())))
        for province in parameters.keys() :
            parameter_arr[get_compartment_id(cities,province)] = sign_rate*parameters[province][parameter_name]

        if (infectious_compartment not in X_eff_dic.keys()) :
            X_eff_dic[infectious_compartment] = make_X_eff(model,Mobility,y,N,infectious_compartment)
        
        NL = S_matrix.dot(parameter_arr*X_eff_dic[infectious_compartment]*np.reciprocal(N))

        # add to final nonlinear term..
        compartment_affected = get_compartment_id(model,process[2])
        nonlinear_terms[compartment_affected::len(model)] += NL
    return nonlinear_terms

class Commuter_model() :
    '''
    Defines the model. Both the ODEs and how we want to solve them.
    '''
    
    def dxdt(self,t,y,R_linear,model,cities,Mobility,N,parameters):
        # t is time
        # y is array of variables (dimension n x 1). y[0+c] is S for compartment c, y[1+c] is I for compartment c.
        # R is linear rate matrix (dimension n x p), has p on entry i,j if variable i is linearly dependent on variable j and rate us p

        # Calculate basic numbers to input in arrays below
        N_populations = len(cities)
        N_compartments = len(model)
        tot_variables = N_populations*N_compartments


        # Define variables
        dy = np.zeros(tot_variables)

        # Insert nonlinear terms
        #for population in range (N_populations) :
        #    dy[population*variables_per_population + 1] = -SI_parameters[population]*y[population*variables_per_population+1]*y[population*variables_per_population+2]
        #    dy[population*variables_per_population + 2] = +SI_parameters[population]*y[population*variables_per_population+1]*y[population*variables_per_population+2]

        # Add linear terms
        dy += R_linear.dot(y)

        # Add nonlinear terms

        Nonlinear_terms = get_nonlinear_terms(model,cities,Mobility,y,N,parameters)
        dy += Nonlinear_terms


        return dy




    def Solve_commuter_model(self,t,y0,R_linear,model,cities,Mobility,N,parameters):

        N_populations = len(cities)
        N_compartments = len(model)

        # Check if initial conditions sum to 1
        #for population in range (N_populations) :
        #    if (np.around(sum(y0[N_compartments*population+1:N_compartments*(population+1)]),decimals=10) != 1) :
        #        print("ERROR: Sum of initial conditions must sum to 1 for every population")
        #        exit()


        t0 = t[0]

        t = t[1:]

        r = ode(self.dxdt)

        # Runge-Kutta with step size control
        r.set_integrator('dopri5')

        # set initial values
        r.set_initial_value(y0,t0)

        # set transmission rate and recovery rate
        r.set_f_params(R_linear,model,cities,Mobility,N,parameters)

        result = np.zeros((N_compartments*N_populations,len(t)+1))
        result[:,0] = y0

        # loop through all demanded time points
        for it, t_ in enumerate(t):
            # get result of ODE integration
            y = r.integrate(t_)

            # write result to result vector
            result[:,it+1] = y

        return result