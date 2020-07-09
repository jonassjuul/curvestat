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

Hdata = {'data-H':np.load('../../curvestat/tests/test_data/data_DK.npy',allow_pickle='TRUE')}
data_time = np.arange(-len(Hdata['data-H']),0,1)

# read simulated curves..
dic_solutions_future = np.load('../../curvestat/tests/test_data/curves_DKE3.npy',allow_pickle='TRUE').item()

# Array of time steps
time_arr = np.arange(0,len(dic_solutions_future[1]),1)



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

# Also find heatmap for all curves. This will be used in figure insets.
heatmap_all = Boxplot.get_peakheatmap(heatmap_curves=list(dic_solutions_future.keys()))


# -----
# GET BOXPLOT: INTERVAL
# -----

# Define the interval of interest
interval_start = 50 # Start at day 200 
interval_end = 100 # End at day 300
interval = []
for i in range ( len(dic_solutions_future[list(dic_solutions_future.keys())[1]]) ) :

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

# Define things for the plot below.

# .. plot this matrix
plot_matrix_valuesduration = load_and_duration

# .. which levels to show in contour plot?
level_vec = [0.01,0.05,0.5]

# .. mesh for x and y axis values.
colormesh_first = np.arange(0,len(plot_matrix_valuesduration[:,0])*10,10)
colormesh_second = np.arange(0,len(plot_matrix_valuesduration[0,:]),1)


















'''
Plotting everything in a single figure
[This code gets messy. You can avoid reading this and instead plot every boxplot using the plot_everything command. (see 'tests' subfolder)]
'''
# MAKE DEFINITIONS FOR PLOT

# Getting numbering of panels in the correct position.....

divide_h = 40

def h_text(xlim,width='narrow') :

    if ( width == 'wide') :
        return (xlim[1] - (xlim[1]-xlim[0])/divide_h)
    
    elif ( width == 'medium') :
        return (xlim[1] - (xlim[1]-xlim[0])/divide_h*1.5)    
        
    else :     
    
        return (xlim[1] - (xlim[1]-xlim[0])/divide_h*3)
def v_text(ylim,width='narrow') :
    
    if ( width == 'wide') :
        return(ylim[1] - (ylim[1]-ylim[0])/7)#10)
    
    else : 
        return(ylim[1] - (ylim[1]-ylim[0])/5)#7)

# convert cm to inches
def cm_to_inch(cm) :
    return 0.3937007874*cm

# Plot this percentile in main frames..
percentile = 50
N_samples = len(dic_solutions_future.keys())



# Import packages for plots
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib import cm


# Fig 2
figsize2 = (cm_to_inch(9),cm_to_inch(14))

fig = plt.figure(figsize=figsize2)

# General: 
plot_dimension = (4,2)
plottime = (np.arange(0,len(boundaries['curve-based'][percentile]['max-boundary']),1))
alpha50 = .6#.25#.50
alpha90 = alpha50/2#.25
hm_scattersize = 2#1.#1.5#2#5
hm_small_scattersize = 1.#1.5#2#5
peak_cmap =  ['viridis','magma','cividis','plasma','inferno'][0]

outside_color = 'C4'
outside_marker = 'v'
outside_alpha=0.8


datamarkersize=1
FT_linewidth = 0.5#1
FT90_linewidth = 0.5
alphaFT90 = 0.5

# INSET NUMBERING
numberingfontsize = 14
halignment = 'right'
valignment='center'

# FONTS
EXTREMELY_SMALL_SIZE = 3.6#4.2
EVEN_SMALLER_SIZE = 5.5
SMALL_SIZE = 7
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# GENERAL SETTINGS
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=EVEN_SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# BOXPLOT PLOT SETTINGS
BP_xticks = (-50,0,50,100,150)#(0,400,800)
BP_xlabel = 'Day'
BP_ylabel = 'Newly Hospitalized'#'Patients hospitalized'
BP_ylim = [-10,850]#[-200,8800]

hide_yticklabels = True
hide_xticklabels = False

show_inset_labels = True
BPinset_ylabel = 'Newly Hosp.'
inset_tickwidth = .4
inset_ticklength = 2
inset_pad =1

horizontal_space_between_subplots = 0#1.
vertical_space_between_subplots = .4#.2

# ---
# TOP: curves
# ---

# define
ax0 = plt.subplot2grid(plot_dimension,(0,0),rowspan=1,colspan=2)

# data
#ax0.plot(np.arange(0,1090,10)/10,Hdata['data-H'],'k.',markersize=2)
ax0.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)

# curves
for sample in (list(dic_solutions_future.keys())[0:200]) :
    ax0.plot(time_arr,dic_solutions_future[sample], c='C0',linewidth=FT_linewidth,alpha=0.2)
# labels
ax0.legend(['Data','Simulations'],frameon=False,loc=2)
ax0.set_xlabel('Day',labelpad=inset_pad)
ax0.set_ylabel('Newly Hospitalized',labelpad=inset_pad)
ax0.xaxis.set_tick_params(pad=inset_pad)
ax0.yaxis.set_tick_params(pad=inset_pad)

xlim = ax0.get_xlim()
ylim = ax0.get_ylim()
ax0.text(h_text(xlim,width='wide'),v_text(ylim,width='wide'),'A',fontsize=numberingfontsize,horizontalalignment=halignment,verticalalignment=valignment)

'''
MIDDLE 1 : 
'''

# ----
# AX1: ALL OR NOTHING
# ----
ax1 = plt.subplot2grid(plot_dimension,(1,0),rowspan=1,colspan=1)

# Plot this dictionary
plot_this_dic = boundaries
ax1.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)
ax1.fill_between(plottime,plot_this_dic['curve-based'][50]['min-boundary'],plot_this_dic['curve-based'][50]['max-boundary'],color='C0',lw=0,alpha=alpha50)

# heatmap
ax1.scatter(heatmap_50['peak_time'], heatmap_50['peak_value'], c=heatmap_50['peak_density'], s=hm_scattersize, edgecolor='',cmap=peak_cmap)

# Fixed-time
ax1.plot(plottime,plot_this_dic['fixed-time'][50]['max-boundary'],'k',linewidth=FT_linewidth)
ax1.plot(plottime,plot_this_dic['fixed-time'][50]['min-boundary'],'k',linewidth=FT_linewidth)


# INSET
axIS1 = inset_axes(ax1, width="40%", height="35%", loc=2)
axIS1.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)

axIS1.fill_between(plottime,plot_this_dic['curve-based'][90]['min-boundary'],plot_this_dic['curve-based'][90]['max-boundary'],color='C0',lw=0,alpha=alpha90,label='CB: 90%')
axIS1.plot(plottime,plot_this_dic['fixed-time'][90]['max-boundary'],'k',linewidth=FT90_linewidth,label='FT: 90%',ls='-',alpha=alphaFT90)
axIS1.plot(plottime,plot_this_dic['fixed-time'][90]['min-boundary'],'k',linewidth=FT90_linewidth,ls='-',alpha=alphaFT90)

if (peak_cmap != 'viridis'):
    axIS1.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C3', s=0.1, edgecolor='',alpha=1)
else : 
    axIS1.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C0', s=0.1, edgecolor='',alpha=1)
axIS1.tick_params(labelsize=EXTREMELY_SMALL_SIZE)
axIS1.set_yticks((0,250,500,750))
axIS1.set_xticks(BP_xticks)
if (show_inset_labels == False):
    axIS1.set_xticklabels([],rotation=0,ha='center')
    axIS1.set_yticklabels([],rotation=0,ha='center')
else:
    axIS1.set_xticklabels([-50,0,50,100,150],rotation=0,ha='center')
    axIS1.set_xlabel(BP_xlabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad = inset_pad)
    axIS1.set_ylabel(BPinset_ylabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)

axIS1.xaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)
axIS1.yaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)    
    
axIS1.yaxis.tick_right()
axIS1.yaxis.set_label_position("right")  

# Labels and other settings
ax1.set_xticks(BP_xticks)
ax1.set_xlabel(BP_xlabel,labelpad=inset_pad)
ax1.set_ylabel(BP_ylabel,labelpad=inset_pad)
ax1.set_ylim(BP_ylim)
ax1.xaxis.set_tick_params(pad=inset_pad)
ax1.yaxis.set_tick_params(pad=inset_pad)

if (hide_xticklabels == True) :
    ax1.axes.xaxis.set_ticklabels([])

xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
ax1.text(h_text(xlim),v_text(ylim),'B',fontsize=numberingfontsize,horizontalalignment=halignment)







# ----
# AX2: ALL OR NOTHING INTERVAL
# ----

ax2 = plt.subplot2grid(plot_dimension,(1,1),rowspan=1,colspan=1)

# Plot this dictionary
plot_this_dic = boundaries_interval
ax2.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)
ax2.fill_between(plottime,plot_this_dic['curve-based'][50]['min-boundary'],plot_this_dic['curve-based'][50]['max-boundary'],color='C0',lw=0,alpha=alpha50)

# heatmap
ax2.scatter(heatmap_50_interval['peak_time'], heatmap_50_interval['peak_value'], c=heatmap_50_interval['peak_density'], s=hm_scattersize, edgecolor='',cmap=peak_cmap)

# Fixed-time
ax2.plot(plottime,plot_this_dic['fixed-time'][50]['max-boundary'],'k',linewidth=FT_linewidth)
ax2.plot(plottime,plot_this_dic['fixed-time'][50]['min-boundary'],'k',linewidth=FT_linewidth)


# INSET
axIS2 = inset_axes(ax2, width="40%", height="35%", loc=2)
axIS2.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)

axIS2.fill_between(plottime,plot_this_dic['curve-based'][90]['min-boundary'],plot_this_dic['curve-based'][90]['max-boundary'],color='C0',lw=0,alpha=alpha90,label='CB: 90%')
axIS2.plot(plottime,plot_this_dic['fixed-time'][90]['max-boundary'],'k',linewidth=FT90_linewidth,label='FT: 90%',ls='-',alpha=alphaFT90)
axIS2.plot(plottime,plot_this_dic['fixed-time'][90]['min-boundary'],'k',linewidth=FT90_linewidth,ls='-',alpha=alphaFT90)


if (peak_cmap != 'viridis'):
    axIS2.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C3', s=0.1, edgecolor='',alpha=1)
else : 
    axIS2.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C0', s=0.1, edgecolor='',alpha=1)
    
axIS2.tick_params(labelsize=EXTREMELY_SMALL_SIZE)
axIS2.set_yticks((0,250,500,750))
axIS2.set_xticks(BP_xticks)
if (show_inset_labels == False):
    axIS2.set_xticklabels([],rotation=0,ha='center')
    axIS2.set_yticklabels([],rotation=0,ha='center')
else:
    axIS2.set_xticklabels([-50,0,50,100,150],rotation=0,ha='center')
    axIS2.set_xlabel(BP_xlabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)
    axIS2.set_ylabel(BPinset_ylabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)

axIS2.xaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)
axIS2.yaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)    
    
axIS2.yaxis.tick_right()
axIS2.yaxis.set_label_position("right")  


# Labels and other settings
ax2.axvline(x=interval_start,ls='--',linewidth=FT_linewidth,c='C1',alpha=0.5)
ax2.axvline(x=interval_end,ls='--',c='C1',linewidth=FT_linewidth,alpha=0.5,label='Interval boundary')

ax2.set_xticks(BP_xticks)
ax2.set_xlabel(BP_xlabel,labelpad=inset_pad)
ax2.set_ylim(BP_ylim)
if (hide_yticklabels == True) :
    ax2.axes.yaxis.set_ticklabels([])
    
if (hide_xticklabels == True) :
    ax2.axes.xaxis.set_ticklabels([])

ax2.xaxis.set_tick_params(pad=inset_pad)
ax2.yaxis.set_tick_params(pad=inset_pad)
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
ax2.text(h_text(xlim),v_text(ylim),'C',fontsize=numberingfontsize,horizontalalignment=halignment)




'''
MIDDLE 2 :
'''


# ----
# AX3: EXPONENTIALLY DECAYING WEIGHTS
# ----

ax3 = plt.subplot2grid(plot_dimension,(2,0),rowspan=1,colspan=1)

# Plot this dictionary
plot_this_dic = boundaries_ExpDecay 
ax3.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)
ax3.fill_between(plottime,plot_this_dic['curve-based'][50]['min-boundary'],plot_this_dic['curve-based'][50]['max-boundary'],color='C0',lw=0,alpha=alpha50)

# heatmap
ax3.scatter(heatmap_50_ExpDecay['peak_time'], heatmap_50_ExpDecay['peak_value'], c=heatmap_50_ExpDecay['peak_density'], s=hm_scattersize, edgecolor='',cmap=peak_cmap)

# Fixed-time
ax3.plot(plottime,plot_this_dic['fixed-time'][50]['max-boundary'],'k',linewidth=FT_linewidth)
ax3.plot(plottime,plot_this_dic['fixed-time'][50]['min-boundary'],'k',linewidth=FT_linewidth)

# INSET
axIS3 = inset_axes(ax3, width="40%", height="35%", loc=2)
axIS3.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)

axIS3.fill_between(plottime,plot_this_dic['curve-based'][90]['min-boundary'],plot_this_dic['curve-based'][90]['max-boundary'],color='C0',lw=0,alpha=alpha90,label='CB: 90%')
axIS3.plot(plottime,plot_this_dic['fixed-time'][90]['max-boundary'],'k',linewidth=FT90_linewidth,label='FT: 90%',ls='-',alpha=alphaFT90)
axIS3.plot(plottime,plot_this_dic['fixed-time'][90]['min-boundary'],'k',linewidth=FT90_linewidth,ls='-',alpha=alphaFT90)
if (peak_cmap != 'viridis'):
    axIS3.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C3', s=0.1, edgecolor='',alpha=1)
else : 
    axIS3.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C0', s=0.1, edgecolor='',alpha=1)
    

axIS3.tick_params(labelsize=EXTREMELY_SMALL_SIZE)
axIS3.set_yticks((0,250,500,750))
axIS3.set_xticks(BP_xticks)
if (show_inset_labels == False):
    axIS3.set_xticklabels([],rotation=0,ha='center')
    axIS3.set_yticklabels([],rotation=0,ha='center')
else:
    axIS3.set_xticklabels([-50,0,50,100,150],rotation=0,ha='center')
    axIS3.set_xlabel(BP_xlabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)
    axIS3.set_ylabel(BPinset_ylabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)

axIS3.xaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)
axIS3.yaxis.set_tick_params(width=inset_tickwidth,length=inset_ticklength,pad=inset_pad)

axIS3.yaxis.tick_right()
axIS3.yaxis.set_label_position("right")    
    

# Labels and other settings

ax3.set_xticks(BP_xticks)
ax3.set_xlabel(BP_xlabel,labelpad=inset_pad)
ax3.set_ylabel(BP_ylabel,labelpad=inset_pad)
ax3.set_ylim(BP_ylim)

ax3.xaxis.set_tick_params(pad=inset_pad)
ax3.yaxis.set_tick_params(pad=inset_pad)

xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
ax3.text(h_text(xlim),v_text(ylim),'D',fontsize=numberingfontsize,horizontalalignment=halignment)




# ----
# AX4 : MAX
# ----
ax4 = plt.subplot2grid(plot_dimension,(2,1),rowspan=1,colspan=1)

# Plot this dictionary
plot_this_dic = boundaries_max
ax4.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)
ax4.fill_between(plottime,plot_this_dic['curve-based'][50]['min-boundary'],plot_this_dic['curve-based'][50]['max-boundary'],color='C0',lw=0,alpha=alpha50,label='CB: 50%')



# Fixed-time
ax4.plot(plottime,plot_this_dic['fixed-time'][50]['max-boundary'],'k',linewidth=FT_linewidth,ls='-',label='FT: 50%')
ax4.plot(plottime,plot_this_dic['fixed-time'][50]['min-boundary'],'k',linewidth=FT_linewidth,ls='-')

# heatmap
ax4.scatter(heatmap_50_max['peak_time'], heatmap_50_max['peak_value'], c=heatmap_50_max['peak_density'], s=hm_scattersize, edgecolor='',cmap=peak_cmap)

# INSET
axIS4 = inset_axes(ax4, width="40%", height="35%", loc=2)
axIS4.plot(data_time,Hdata['data-H'],'k.',markersize=datamarkersize)

axIS4.fill_between(plottime,plot_this_dic['curve-based'][90]['min-boundary'],plot_this_dic['curve-based'][90]['max-boundary'],color='C0',lw=0,alpha=alpha90,label='CB: 90%')
axIS4.plot(plottime,plot_this_dic['fixed-time'][90]['max-boundary'],'k',linewidth=FT90_linewidth,label='FT: 90%',ls='-',alpha=alphaFT90)
axIS4.plot(plottime,plot_this_dic['fixed-time'][90]['min-boundary'],'k',linewidth=FT90_linewidth,ls='-',alpha=alphaFT90)

if (peak_cmap != 'viridis'):
    axIS4.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C3', s=0.1, edgecolor='',alpha=1)
else : 
    axIS4.scatter(heatmap_all['peak_time'], heatmap_all['peak_value'], c='C0', s=0.1, edgecolor='',alpha=1)
    

axIS4.tick_params(labelsize=EXTREMELY_SMALL_SIZE)
axIS4.set_yticks((0,250,500,750))
axIS4.set_xticks(BP_xticks)
if (show_inset_labels == False):
    axIS4.set_xticklabels([],rotation=0,ha='center')
    axIS4.set_yticklabels([],rotation=0,ha='center')
else:
    axIS4.set_xticklabels([-50,0,50,100,150],rotation=0,ha='center')
    axIS4.set_xlabel(BP_xlabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)
    axIS4.set_ylabel(BPinset_ylabel,fontsize=EXTREMELY_SMALL_SIZE,labelpad=inset_pad)

axIS4.xaxis.set_tick_params(width=.4,length=2,pad=inset_pad)
axIS4.yaxis.set_tick_params(width=.4,length=2,pad=inset_pad)

axIS4.yaxis.tick_right()
axIS4.yaxis.set_label_position("right")


# Labels and other settings

ax4.set_xticks(BP_xticks)
ax4.xaxis.set_tick_params(pad=inset_pad)
ax4.set_xlabel(BP_xlabel,labelpad=inset_pad)
ax4.set_ylim(BP_ylim)
ax4.xaxis.set_tick_params(pad=inset_pad)
ax4.yaxis.set_tick_params(pad=inset_pad)

if (hide_yticklabels == True) :
    ax4.axes.yaxis.set_ticklabels([])

xlim = ax4.get_xlim()
ylim = ax4.get_ylim()
ax4.text(h_text(xlim),v_text(ylim),'E',fontsize=numberingfontsize,horizontalalignment=halignment)

ax4.legend(loc=3,frameon=False)


'''
BOTTOM
'''



# ----
# AX6: COLORMAP
# ----
ax6 = plt.subplot2grid(plot_dimension,(3,0),rowspan=1,colspan=2)

meshax6 = ax6.pcolormesh(colormesh_second,colormesh_first,plot_matrix_valuesduration,cmap=cm.Blues)
cbar = plt.colorbar(meshax6)
cbar.set_label('Probability',labelpad=inset_pad)

ax6.set_xlabel('Duration [days]',labelpad=inset_pad)
ax6.set_ylabel(BP_ylabel,labelpad=inset_pad)

CS = ax6.contour(colormesh_second,colormesh_first,plot_matrix_valuesduration,levels=level_vec,labels=level_vec,colors='k',linewidths=FT_linewidth,alpha=0.8)
ax6.clabel(CS,CS.levels, inline=True, fmt='%2.2f',fontsize=SMALL_SIZE)

ax6.set_ylim(BP_ylim)
ax6.xaxis.set_tick_params(pad=inset_pad)
ax6.yaxis.set_tick_params(pad=inset_pad)
cbar.ax.xaxis.set_tick_params(pad=inset_pad)

xlim = ax6.get_xlim()
ylim = ax6.get_ylim()
ax6.text(h_text(xlim,width='medium'),v_text(ylim),'F',fontsize=numberingfontsize,horizontalalignment=halignment)


fig.align_ylabels([ax0,ax1,ax3,ax6])

plt.tight_layout(h_pad=vertical_space_between_subplots)
plt.savefig('Figures/Fig2_onecolumn_DKinsets_%s.png'%peak_cmap,dpi=400)