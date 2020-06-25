'''
Returns curve-based summary statistics 
'''

import random
import numpy as np 
import copy
from scipy.stats import gaussian_kde

class CurveBoxPlot() :
    
    def __init__(self,curves,sample_curves=10,sample_repititions=100,interval=None,weights=None,time=None,plot_percentiles=None,plot_heatmap=None,verbose=False) :
        '''
        Parameters
        ----------

        curves : :obj: `dict` of arrays
            a dict that contains the curves which we would
            like to create the curve-based descriptive 
            statistics for.

        sample_curves : :int:

        sample_repititions : :int:

        ranking : :obj: `dict` with {curve_name:score}
            higher score means more central.

        percentiles : :obj: `list` with percentiles we want to find.
            List entry X means we want to find the X _most central_ curves.
            Example: percentiels = [50] find the most central 50% of curves,
            i.e. the 25-75 percentiles.

        interval : :obj: `list` with bool entries.
            Entry i is True if i is in interval. False otherwise.

        weights : 
            Specifies reward for a curve falling inside an
            envelope at a time step.

        verbose : bool, default: False



        '''

        self.curves = copy.deepcopy(curves)
        self.sample_parameters = {'N_curves':sample_curves,'N_repititions':sample_repititions}
        #self.ranking = ranking
        #self.percentiles = percentiles        
        self.interval = interval
        self.weights = weights
        self.time = time
        #self.heatmap_curves = heatmap_curves
        self.plot_percentiles = plot_percentiles
        self.plot_heatmap = plot_heatmap

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
                curve_copies[curve] = np.array(curve_copies[curve][interval])
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

        return percentiles_results

    def get_peakheatmap(self,heatmap_curves) :
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
        
        return peak_heatmap


    def plot_everything(self) :
        fig, ax = plt.subplots()

        if (self.plot_percentiles != None) :
            # First, plot Curve-based percentiles,

            # Find number of minimum percentile. This is useful when defining which spaces to fill with color
            number_of_percentiles = len(self.plot_percentiles['curve-based'].keys())
            percentiles_by_value = sorted(self.plot_percentiles['curve-based'].keys())

            # Plot least extreme area
            ax.fill_between(self.time,self.plot_percentiles['curve-based'][percentiles_by_value[0]]['min-boundary'],self.plot_percentiles['curve-based'][percentiles_by_value[0]]['max-boundary'],color='C0',alpha=0.5)

            for percentile in range (len(percentiles_by_value)-1) :
                ax.fill_between(self.time,self.plot_percentiles['curve-based'][percentiles_by_value[percentile]]['max-boundary'],self.plot_percentiles['curve-based'][percentiles_by_value[percentile+1]]['max-boundary'],color='C0',alpha=0.5/(percentile+2))
                ax.fill_between(self.time,self.plot_percentiles['curve-based'][percentiles_by_value[percentile]]['min-boundary'],self.plot_percentiles['curve-based'][percentiles_by_value[percentile+1]]['min-boundary'],color='C0',alpha=0.5/(percentile+2))

            # Then, plot Fixed-time percentiles
            percentile_number = -1
            for percentile in self.plot_percentiles['fixed-time'].keys() :
                percentile_number +=1
                ax.plot(self.time,self.plot_percentiles['fixed-time'][percentile]['min-boundary'],'k',alpha=1/(1+percentile_number))
                ax.plot(self.time,self.plot_percentiles['fixed-time'][percentile]['max-boundary'],'k',alpha=1/(1+percentile_number))

        if (self.plot_heatmap) :
            ax.scatter(self.plot_heatmap['peak_time'], self.plot_heatmap['peak_value'], c=self.plot_heatmap['peak_density'], s=50, edgecolor='')


class LoadRisk() :

    def __init__(self,curves,verbose=False) :

        '''
        Parameters
        
        curves:

        verbose: 

        '''

        self.curves = curves
        self.verbose = verbose

    def get_loadandduration(self) :

        '''
        Creates matrix to be plotted as heatmap. 
        Dimension are time and load and entry (i,j) is the fraction of curves that exceed the value i for at least a duration j.
        '''
        load_granularity = 10
        time_granularity = 1


        curve_copies = copy.deepcopy(self.curves)

        # Find maximum value of any curve in the ensemble
        max_load = max_value_in_ensemble(curve_copies)

        # Define dimensions of result matrix
        max_load_dimension = int(max_load/10)+10
        max_time_dimension = len(curve_copies[list(curve_copies.keys())[0]])

        # Array for results..
        colormap_loadduration = np.zeros((max_load_dimension,max_time_dimension))           

        # Loop through curves. Find duration of load.
        sample_number = -1
        for sample in (curve_copies.keys()) :
            sample_number +=1 
            if (self.verbose == True) :
                if (sample_number/100 == sample_number // 100) :
                    print("Doing curve number",sample_number,"out of",len(curve_copies.keys()))
            this_curve = curve_copies[sample]


            for load_entry in range (int(max(this_curve)/load_granularity)+1) :
                load = load_entry*load_granularity+0
                
                this_curve = this_curve[this_curve>=load]

                time_above = len(this_curve)
                entries_above = int(time_above/time_granularity)
                
                if (entries_above == 0) :
                    break

                # save this in array
                colormap_loadduration[load_entry,:entries_above+1] += 1            


        colormap_loadduration /= len(curve_copies.keys())
        return colormap_loadduration


    #def plot_heatmap(self,colormap_matrix):



def max_value_in_ensemble(curves) :
    max_value = 0
    for sample in (curves.keys()) :
        this_max = max(curves[sample])
        if ( this_max > max_value) :
            max_value = this_max + 0
    return max_value

