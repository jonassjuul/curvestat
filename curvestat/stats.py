'''
Returns curve-based summary statistics 
'''

import random
import numpy as np 
import copy
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt 
from matplotlib import cm

class CurveBoxPlot() :
    
    def __init__(self,curves,sample_curves=10,sample_repititions=100,interval=None,time_weights=None,curve_weights=None,time=None,verbose=False) :
        '''
        Class to make curve box plots.


        Parameters
        ----------

        curves : :obj: `dict` of arrays
            a dict that contains the curves which we would
            like to create the curve-based descriptive 
            statistics for.

        sample_curves : :int:

        sample_repititions : :int:

        interval : :obj: `list` with bool entries.
            Entry i is True if i is in interval. False otherwise.

        time_weights : :obj: `list` of floats
            Specifies reward for a curve falling inside an
            envelope at a time step.

        curve_weights : `dict` with same keys as 'curves'. values are :int:
            Specifies factor that the score of a curve will be
            multiplied with before the ranking is returned. 
            Factor could be dependent on prior of curves.

        time : :obj: `array` of same length as the arrays in curves.
            Array elements are the time steps corresponding to curve entries. 

        verbose : bool, default: False
            Print stuff?

        '''

        self.curves = copy.deepcopy(curves)
        self.sample_parameters = {'N_curves':sample_curves,'N_repititions':sample_repititions}
        self.interval = interval
        self.weights = time_weights
        self.curve_weights = curve_weights
        self.time = time

    def rank_allornothing(self) :

        '''
        Ranks curve in an "all-or-nothing" manner. 
        1) Draws random curves and finds envelope of these.
        2) Gives curve j 1 point if it is entirely contained in envelope in interval of interest.
        3) Repeats many times
        4) Returns resulting ranking. Most points = highest ranking.
        '''

        curve_copies = copy.deepcopy(self.curves)

        # get keys of curves in ensemble
        curve_names = list(self.curves.keys())

        if (self.interval != None) :

            # If specified interval, only take this interval into account when ranking.
            for curve in curve_names :
                # This discards everything but interval from the analysis
                curve_copies[curve] = np.array(curve_copies[curve][self.interval])
        else :
            # If no specified interval, get all curves as numpy arrays
            for curve in curve_names :
                curve_copies[curve] = np.array(curve_copies[curve])


        # reset dic with curve scores [more scores = more central in the ensemble]
        curve_scores = {}
        for curve in curve_names :
            curve_scores[curve] = 0

        # Now sample random curves a number of times
        for sample_number in range (self.sample_parameters['N_repititions']) :
            
            # Pick random sample
            curve_names_in_this_sample = random.sample(curve_names, self.sample_parameters['N_curves'])

            # Save curves from the random sample in an array
            curve_collection = []
            for curve in curve_names_in_this_sample : 
                curve_collection.append(curve_copies[curve])

            # Find envelope of the random sample
            max_boundary = np.maximum.reduce(curve_collection)
            min_boundary = np.minimum.reduce(curve_collection)

            # Check which curves are entirely inside the envelope and reward them 1 point.
            for curve in curve_names :

                is_above_max = np.sum(curve_copies[curve]>max_boundary)
                if ( is_above_max == 0) :
                    
                    is_below_min = np.sum(curve_copies[curve] < min_boundary)

                    if ( is_below_min == 0 ) :
                        # Reward with 1 point
                        curve_scores[curve] += 1

        if (self.curve_weights != None) :
            for curve in curve_names :
                curve_scores[curve] *= self.curve_weights[curve]

        return curve_scores

    def rank_weighted(self) :

        '''
        Ranks curves given weights. Weight i is the reward a curve receives if it lies within the random envelope at time step i.
        '''

        # get keys of curves in ensemble
        curve_names = list(self.curves.keys())

        # reset dic with curve scores [more scores = more central in the ensemble]
        curve_scores = {}
        for curve in curve_names :
            curve_scores[curve] = 0

        # Now sample random curves a number of times
        for sample_number in range (self.sample_parameters['N_repititions']) :
            
            # Pick random sample
            curve_names_in_this_sample = random.sample(curve_names, self.sample_parameters['N_curves'])

            # Save curves from the random sample in an array
            curve_collection = []
            for curve in curve_names_in_this_sample : 
                curve_collection.append(self.curves[curve])

            # Find envelope of the random sample
            max_boundary = np.maximum.reduce(curve_collection)
            min_boundary = np.minimum.reduce(curve_collection)


            # Check which curves are entirely inside the envelope and reward them point as specified by weights.
            # we assume that 'weights' has 0 on entry i, if i should not be taken into consideration.
            for curve in curve_names :
                is_above_max = np.sum(np.multiply(self.curves[curve]>max_boundary,self.weights))
                    
                is_below_min = np.sum(np.multiply(self.curves[curve] < min_boundary,self.weights))

                # Reward with points for all entries not below min or above max.
                curve_scores[curve] += np.sum(weights)-is_above_max-is_below_min


        if (self.curve_weights != None) :
            for curve in curve_names :
                curve_scores[curve] *= self.curve_weights[curve]


        return curve_scores        

    def rank_max(self) :

        '''
        Ranks curves according to max value. Median has highest score.
        '''

        # get keys of curves in ensemble
        curve_names = list(self.curves.keys())

        # find max of all curves
        curve_max = {}
        for curve in curve_names :

            curve_max[curve] = max(self.curves[curve])+0

        # sort dictionary curve_max according to value
        curve_max_sorted = {k: v for k, v in sorted(curve_max.items(), key=lambda item: item[1])}

        # Reward median max most. Extremes the least. Here we do it using a second degree polynomial: curve with ith highest max gets -i*(i-(N-1)) points.
        sequence = np.arange(0,len(curve_max),1)
        curve_scores_array = -np.multiply(sequence,sequence-(len(curve_max)-1))


        # Match curve scores with curve names and return dictionary.
        curve_scores = {}
        curve_number = -1
        for curve in curve_max_sorted.keys() :
            curve_number +=1
            curve_scores[curve] = curve_scores_array[curve_number]

        return curve_scores


    def get_boundaries(self,ranking,percentiles) :

        '''
        Calculates curve-based and fixed-time percentiles from ranking.
        
        INPUTS
        ----------
        ranking : :obj: `dict` with {curve_name:score}
            higher score means more central.        

        percentiles : :obj: `list` with percentiles we want to find.
            List entry X means we want to find the X _most central_ curves.
            Example: percentiles = [50] find the most central 50% of curves,
            i.e. the 25-75 percentiles.            

        
        '''

        self.ranking = ranking
        self.percentiles = percentiles

        # get keys of curves in ensemble
        curve_names = list(self.curves.keys())

        # Define dictionary for results
        percentiles_results= {}

        # ----
        # First: Get curve-based percentiles
        # ----
        percentiles_results['curve-based'] = {}

        # Sort ranking. Most central on top.
        sorted_ranking = sorted(list(self.ranking.values()),reverse=True)

        for percentile in self.percentiles :

            # 
            percentiles_results['curve-based'][percentile] = {'curves':[]} 

            # Find rank-value that separates curves we want from the rest.
            threshold_value = sorted_ranking[int(len(sorted_ranking)*percentile/100)]

            # determine which curves are in wanted interval
            curve_collection = []
            for curve in self.ranking.keys() :
                # Check if score is above threshold
                if (self.ranking[curve]>=threshold_value) :
                    # Then append to list of curves
                    curve_collection.append(self.curves[curve])

                    # and save curve name to results
                    percentiles_results['curve-based'][percentile]['curves'].append(curve)

            # Save boundaries of percentile..
            percentiles_results['curve-based'][percentile]['max-boundary'] = np.maximum.reduce(curve_collection)
            percentiles_results['curve-based'][percentile]['min-boundary'] = np.minimum.reduce(curve_collection)

        # ----
        # Second: Get Fixed-Time percentiles
        # ----
        percentiles_results['fixed-time'] = {}

        for percentile in self.percentiles : 
            percentiles_results['fixed-time'][percentile] = {'max-boundary':[],'min-boundary':[]}

            # What percentiles are we actually finding? (e.g., convert 50 most central -> 25% and 75% percentiles)
            max_boundary = 50+percentile/2
            min_boundary = 50-percentile/2


            last_timestep = False
            timestep = -1
            while (last_timestep == False) :
                # Make array to store value on current time step

                array_values_this_time_step = []

                timestep +=1
                for curve in curve_names :
                    array_values_this_time_step.append(self.curves[curve][timestep])

                # Now sort values, smallest on top.
                array_values_this_time_step.sort()

                percentiles_results['fixed-time'][percentile]['max-boundary'].append(array_values_this_time_step[int(len(array_values_this_time_step)*max_boundary/100)]+0)
                percentiles_results['fixed-time'][percentile]['min-boundary'].append(array_values_this_time_step[int(len(array_values_this_time_step)*min_boundary/100)]+0)


                if (len(self.curves[curve]) == timestep+1) :
                    last_timestep = True

        # Save and results
        self.percentiles_results = percentiles_results
        return percentiles_results

    def get_peakheatmap(self,heatmap_curves) :


        '''
        Finds time and values of curve peaks for curves
        listed in 'heatmap_curves'. Then uses a gaussian
        kernel to assign each a value indicating density
        of peaks around this peak. 


        INPUTS
        ----------
        heatmap_curves : :obj: `list` keys from 'curves' dict. 
            These are the curves that the peaks will be found
            for.
        '''

        self.heatmap_curves = heatmap_curves

        # find peaks
        peak_time = []
        peak_value = []

        for curve in self.heatmap_curves :
            peak_value.append(max(self.curves[curve]))
            peak_time.append(self.time[self.curves[curve]==peak_value[-1]][0])

        # Get x and y-coordinates
        peak_time = np.array(peak_time)
        peak_value = np.array(peak_value)

        # Calculate point density
        xy = np.vstack([peak_time,peak_value])
        peak_density = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = peak_density.argsort()
        peak_heatmap = {'peak_time':peak_time[idx], 'peak_value':peak_value[idx],'peak_density': peak_density[idx]}
        
        # Save heatmap
        self.heatmap = peak_heatmap

        return peak_heatmap


    def plot_everything(self) :

        '''
        Plots all curve-based and fixed-time percentiles for box plot class.
        Also plots peak heatmap and interval boundaries if such are defined
        for the box plot.
        '''

        fig, ax = plt.subplots()

        if (self.percentiles != None) :
            # First, plot Curve-based percentiles,

            # Find number of minimum percentile. This is useful when defining which spaces to fill with color
            number_of_percentiles = len(self.percentiles)
            percentiles_by_value = sorted(self.percentiles)


            # Plot least extreme area
            ax.fill_between(self.time,self.percentiles_results['curve-based'][percentiles_by_value[0]]['min-boundary'],self.percentiles_results['curve-based'][percentiles_by_value[0]]['max-boundary'],color='C0',alpha=0.5,label='CB: %s'%percentiles_by_value[0])

            for percentile in range (len(percentiles_by_value)-1) :
                ax.fill_between(self.time,self.percentiles_results['curve-based'][percentiles_by_value[percentile]]['max-boundary'],self.percentiles_results['curve-based'][percentiles_by_value[percentile+1]]['max-boundary'],color='C0',alpha=0.5/(percentile+2),label='CB: %s'%percentiles_by_value[percentile+1])
                ax.fill_between(self.time,self.percentiles_results['curve-based'][percentiles_by_value[percentile]]['min-boundary'],self.percentiles_results['curve-based'][percentiles_by_value[percentile+1]]['min-boundary'],color='C0',alpha=0.5/(percentile+2))

            # Then, plot Fixed-time percentiles
            percentile_number = -1
            for percentile in self.percentiles_results['fixed-time'].keys() :
                percentile_number +=1
                ax.plot(self.time,self.percentiles_results['fixed-time'][percentile]['min-boundary'],'k',alpha=1/(1+percentile_number),label='FT: %s'%percentile)
                ax.plot(self.time,self.percentiles_results['fixed-time'][percentile]['max-boundary'],'k',alpha=1/(1+percentile_number))

        if (self.interval) :
            
            bool_reverse = np.invert(self.interval)
            interval_boundaries = self.interval[1:]*bool_reverse[0:-1]+self.interval[0:-1]*bool_reverse[1:]
            boundary_times = self.time[1:][interval_boundaries]

            for boundary_time in boundary_times[:-1] :
                ax.axvline(x=boundary_time,ls='--',c='C1',alpha=0.5)
            boundary_time = boundary_times[-1]
            ax.axvline(x=boundary_time,ls='--',c='C1',alpha=0.5,label='Interval')            

        if (self.heatmap) :

            ax.scatter(self.heatmap['peak_time'], self.heatmap['peak_value'], c=self.heatmap['peak_density'], s=50)#, edgecolor='')


class LoadRisk() :

    def __init__(self,curves,verbose=False) :
        '''
        Class to make heatmap of load vs duration.


        PARAMETERS
        ------------

        curves : :obj: `dict` of arrays
            a dict that contains the curves which we would
            like to create the curve-based descriptive 
            statistics for.

        verbose : bool, default: False
            Print stuff?
        '''

        self.curves = curves
        self.verbose = verbose

    def get_loadandduration(self,load_granularity = 10,time_granularity = 1) :

        '''
        Creates matrix to be plotted as heatmap. 
        Dimension are time and load and entry (i,j) is the fraction of curves that exceed the value i for at least a duration j.

        PARAMETERS
        ------------------
        load_granularity :int:
            integer indicating granularity of heatmap. The function will check how many curve entries show values above 
            0, load_granularity, 2*load_granularity,... 
        
        time_granulairity :int:

        '''

        # Granularity of analysis. If load_granularity is 10, we check how many curves have more than 0,10,20,30,... patients hospitalized for duration of interest.
        self.load_granularity =  load_granularity
        self.time_granularity = time_granularity


        curve_copies = copy.deepcopy(self.curves)

        # Find maximum value of any curve in the ensemble
        max_load = max_value_in_ensemble(curve_copies)

        # Define dimensions of result matrix
        max_load_dimension = int(max_load/load_granularity)+load_granularity
        max_time_dimension = int(len(curve_copies[list(curve_copies.keys())[0]])/time_granularity)+time_granularity

        # Array for results..
        colormap_loadduration = np.zeros((max_load_dimension,max_time_dimension))           

        # Loop through curves. Find duration of load.
        sample_number = -1
        for sample in (curve_copies.keys()) :
            sample_number +=1 
            
            # Choose single curve
            this_curve = curve_copies[sample]

            # Iterate over load entries.
            for load_entry in range (int(max(this_curve)/load_granularity)+1) :
                load = load_entry*load_granularity+0
                
                # How many curve entries show >= load patients. Remove the rest of the entries.
                this_curve = this_curve[this_curve>=load]

                # Number of result-matrix time entries that are above the load.
                time_above = len(this_curve)
                entries_above = int(time_above/time_granularity)
                
                if (entries_above == 0) :
                    break

                # save this in array
                colormap_loadduration[load_entry,:entries_above+1] += 1            

        # Normalize each entry to get probability
        colormap_loadduration /= len(curve_copies.keys())

        # Save result
        self.colormap_loadduration = colormap_loadduration

        return colormap_loadduration

    def plot_everything(self,levels=[0.05,0.1,0.3,0.5,0.8]) :

        '''
        Plots heatmap of load vs duration.
        '''
        
        # mesh for colormap (multiplying with granularity to get axis units right)
        colormesh_first = np.arange(0,len(self.colormap_loadduration[:,0])*self.load_granularity,self.load_granularity)
        colormesh_second = np.arange(0,len(self.colormap_loadduration[0,:])*self.time_granularity,self.time_granularity)

        # Plot
        fig, ax = plt.subplots()

        # Heatmap
        meshax = ax.pcolormesh(colormesh_second,colormesh_first,self.colormap_loadduration,cmap=cm.Blues)
        
        # Colormap
        cbar = plt.colorbar(meshax)
        cbar.set_label('Probability')

        # Contours
        CS = ax.contour(colormesh_second,colormesh_first,self.colormap_loadduration,levels=levels,colors='k',linewidths=1,alpha=0.8)
        ax.clabel(CS,CS.levels, inline=True, fmt='%2.2f')



def max_value_in_ensemble(curves) :
    '''
    Find the total maximum value of ensemble of curves.
    
    PARAMETERS
    -----------
    curves : :obj: `dict` of arrays
        a dict that contains the curves which we would
        like to create the curve-based descriptive 
        statistics for.    
    '''
    max_value = 0
    for sample in (curves.keys()) :
        this_max = max(curves[sample])
        if ( this_max > max_value) :
            max_value = this_max + 0
    return max_value

