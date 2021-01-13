# Curvestat
## About

Compute and plot curve-based summary statistics for ensembles of curves. `curvestat` was written by [Jonas L. Juul](http://www.nbi.dk/~jonassj/).


## Paper

If you use `curvestat` please consider citing us! [The paper is available here](www.nature.com/articles/s41567-020-01121-y). [Download citation](https://www.nature.com/articles/s41567-020-01121-y.ris)

## Install

    pip install curvestat

`curvestat` was developed and tested for 

* Python 3.7

## What can you do with `curvestat`?
Given some ensemble of curves, 

<img src="https://github.com/jonassjuul/curvestat/blob/master/paper/code_for_figures/other_images/simulations.png" alt="curve box plot with all-or-nothing ranking" width="50%" height="50%">

`curvestat` lets you compute and visualize curve box plots and compare them to fixed-time descriptive statistics.

<img src="https://github.com/jonassjuul/curvestat/blob/master/curvestat/tests/test_outputs/all_or_nothing_full.png" alt="curve box plot with all-or-nothing ranking" width="50%" height="50%">

`curvestat` also lets you make load-duration heatmaps. In the following heatmap for example, the color at (x,y) illustrates the risk of experiencing a load of at least y patients for at least x consecutive days, given an ensemble of 2000 simulated epidemic trajectories.

<img src="https://github.com/jonassjuul/curvestat/blob/master/curvestat/tests/test_outputs/colormap_LoadandDuration.png" alt="heatmap where color at (x,y) illustrates risk of experiencing a load of at least y patients for at least x consecutive days" width="50%" height="50%">

## Examples
For examples, please see <a href="https://github.com/jonassjuul/curvestat/tree/master/curvestat/tests"> the 'tests' subfolder </a>

## License

This project is licensed under the [MIT License](https://github.com/jonassjuul/curvestat/curvestat/blob/master/LICENSE).
