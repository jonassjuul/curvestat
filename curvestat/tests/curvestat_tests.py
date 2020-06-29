import numpy as np 
import matplotlib.pyplot as plt 
import copy

from curvestat import CurveBoxPlot
from curvestat import LoadRisk

'''
-------------
DESCRIPTION 
-------------

Here we import an example ensemble of curves.

Then we make 4 different kinds of curve box plots using the following rankings
    - All-or-nothing
    - All-or-nothing in a specific time interval
    - Ranking on curve max values
    - Weighted ranking with weights decaying exponentially with time

Given the ensemble of curves showing hospitalized patients as a function of time, 
we also make a heatmap showing the fraction of curves that have x consecutive days 
with a load of at least y patients.
'''



'''
DATA
'''
# Load data points
Hdata = np.load('test_data/datapoints_compartmental_simulation.npy',allow_pickle='TRUE').item()
data_time = np.arange(0,len(Hdata['data-H']),1)

# read simulated curves..
dic_solutions = np.load('test_data/curves_compartmental_simulation.npy',allow_pickle='TRUE').item()

# Get dic of future values for simulations
dic_solutions_future = {}
last_day_of_data = 110
for curve in dic_solutions.keys() :
    # Original dict has 10 values per day. In new dictionary, take only 1 value per day.
    dic_solutions_future[curve] = copy.copy(dic_solutions[curve][last_day_of_data:])

# Array of time steps
time_arr = np.arange(0,len(dic_solutions_future[0]),1)+last_day_of_data



'''
CURVE BOX PLOTS
'''

# -----
# GET BOXPLOT: ALL OR NOTHING
# -----

# ... ENVELOPE :

# define boxplot.
Boxplot = CurveBoxPlot(curves=dic_solutions_future,sample_curves=50,sample_repititions=100,time=time_arr)

# Choose ranking
rank_allornothing=Boxplot.rank_allornothing()

# Get envelope with this choice of ranking
boundaries = Boxplot.get_boundaries(rank_allornothing,percentiles=[50,90])


# ... HEATMAP : 
# First, define which curves we want the heatmap for
heatmapcurves = list(boundaries['curve-based'][50]['curves'])

# Then find the heatmap
heatmap_50 = Boxplot.get_peakheatmap(heatmap_curves=heatmapcurves)

# Plot results
Boxplot.plot_everything()
# Plot data
plt.plot(data_time,Hdata['data-H'],'k.',label='data')

# Legend, labels and save result.
plt.legend(loc=1,frameon=False)
plt.xlabel('Day')
plt.ylabel('Hospitalized')
plt.tight_layout()
plt.savefig('test_outputs/all_or_nothing_full.png',dpi=400)


# -----
# GET BOXPLOT: INTERVAL
# -----

# Define the interval of interest
interval_start = 200 # Start at day 200 
interval_end = 300 # End at day 300
interval = []
for i in range ( len(dic_solutions_future[list(dic_solutions_future.keys())[0]]) ) :

    if (time_arr[i]> interval_start and time_arr[i] < interval_end) :
        interval.append(True)
    else : interval.append(False)
        
# Now do same as above (get boxplots, plot everything):

# define boxplot.
Boxplot_interval = CurveBoxPlot(curves=dic_solutions_future,sample_curves=10,sample_repititions=100,interval=interval,time=time_arr)

# Choose ranking
rank_allornothing_interval =Boxplot_interval.rank_allornothing()

# Get envelope
boundaries_interval = Boxplot_interval.get_boundaries(rank_allornothing_interval,percentiles=[50,90])

# Get heatmap
heatmapcurves_interval = list(boundaries_interval['curve-based'][50]['curves'])
heatmap_50_interval = Boxplot_interval.get_peakheatmap(heatmap_curves=heatmapcurves_interval)

# Plot results
Boxplot_interval.plot_everything()
# Plot data
plt.plot(data_time,Hdata['data-H'],'k.',label='data')

# Legend, labels and save result.
plt.legend(loc=1,frameon=False)
plt.xlabel('Day')
plt.ylabel('Hospitalized')
plt.tight_layout()
plt.savefig('test_outputs/all_or_nothing_interval.png',dpi=400)



# ----
# GET BOXPLOT: RANK BY MAX
# ----

# Define boxplot
Boxplot_max = CurveBoxPlot(curves=dic_solutions_future,sample_curves=50,sample_repititions=100,time=time_arr)

# Choose ranking
rank_max = Boxplot_max.rank_max()

# Get envelope
boundaries_max = Boxplot_max.get_boundaries(rank_max,percentiles=[50,90])

# Get heatmap
heatmapcurves_max = list(boundaries_max['curve-based'][50]['curves'])
heatmap_50_max = Boxplot_max.get_peakheatmap(heatmap_curves=heatmapcurves_max)

# Plot results
Boxplot_max.plot_everything()
# Plot data
plt.plot(data_time,Hdata['data-H'],'k.',label='data')

# Legend, labels and save result.
plt.legend(loc=1,frameon=False)
plt.xlabel('Day')
plt.ylabel('Hospitalized')
plt.tight_layout()
plt.savefig('test_outputs/all_or_nothing_max.png',dpi=400)


# ----
# GET BOXPLOT: EXPONENTIAL DECAY
# ----

# Define weights
def exp_decaying(time_vec,halftime) :
    time_vec = np.array(time_vec)
    return np.exp(-(time_vec-time_vec[0])*np.log(2)/halftime)

halftime = 7 #days
weights = exp_decaying(time_arr,halftime) #divide by 10 to make halftime right


# Define boxplot
Boxplot_ExpDecay = CurveBoxPlot(curves=dic_solutions_future,sample_curves=50,sample_repititions=100,time_weights=weights,time=time_arr)

# Choose ranking
rank_ExpDecay=Boxplot_ExpDecay.rank_allornothing()

# Get envelope
boundaries_ExpDecay = Boxplot_ExpDecay.get_boundaries(rank_ExpDecay,percentiles=[50,90])

# Get heatmap
heatmapcurves_ExpDecay = list(boundaries_ExpDecay['curve-based'][50]['curves'])
heatmap_50_ExpDecay = Boxplot_ExpDecay.get_peakheatmap(heatmap_curves=heatmapcurves_ExpDecay)

# Plot results
Boxplot_ExpDecay.plot_everything()
# Plot data
plt.plot(data_time,Hdata['data-H'],'k.',label='data')

# Legend, labels and save result.
plt.legend(loc=1,frameon=False)
plt.xlabel('Day')
plt.ylabel('Hospitalized')
plt.tight_layout()
plt.savefig('test_outputs/all_or_nothing_ExpDecay.png',dpi=400)

'''
RISK ASSESSMENT: LOAD VS DURATION
'''

# ----
# GET COLOR MAP: LOAD / DURATION
# ----

# Define class
LR = LoadRisk(curves=dic_solutions_future,verbose = True)

# Get matrix with results
load_and_duration = LR.get_loadandduration()

LR.plot_everything()


# Labels and save result
plt.xlabel('Duration [days]')
plt.ylabel('Hospitalized')
plt.tight_layout()
plt.savefig('test_outputs/colormap_LoadandDuration.png',dpi=400)

