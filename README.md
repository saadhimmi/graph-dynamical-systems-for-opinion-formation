# Graph dynamical systems for opinion formation 

For details on the project, methodology and results, please refer to the report.

## Authors 
- Group name : Scotland Hippos
- Group members : Robin Bölsterli, Leonardo Di Felice, Saad Himmi and Marcel Müller

## Reproducibility

After cloning the repository or downloading the source code, you can generate your own results. Here is the syntax to use : 

```
python ./main.py MODEL
```

By MODEL we actually mean one of these strings  : {TEACHER, DIPLOMAT, CIRCLE}. 

*main.py* runs the weight optimization for MODEL, the social learning task and saves 3 figures (that you can find in our report) and a *.npy* file containing all the variables computed during the run (for further use, optimization or visualization). 

Note : We suggest using main.py only on TEACHER. For DIPLOMAT and CIRCLE, because they have more parameters to optimize, you should use *main_parallel.py*. This script actually runs the optimization weight task on 10% of the data : the x in MODEL (from 0 to 9) denotes which subarray it will optimize (eg. DIPLOMAT5, or DIPLOMAT9). Modifying the code this way allowed us to run the optimization task on 10 parallel nodes and save precious time. *main_parallel.py* outputs a single .npy file with the intermediate computations done.

To complete the simulation or save further figures, the single npy file (for main.py) or the 10 intermediate npy files (for main_parallel.py) have to be placed in a single, eponym, directory (ie. CIRCLE) and then run : 

```
python ./main_viz.py MODEL
```

Please refer to the source code for further details on the implementation : the reader might have to rename some files to make the parallel computation work. 
