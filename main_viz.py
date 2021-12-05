import os, sys
import numpy as np
import itertools
from experimental_design import create_experiment

def main():
    # Deal with directories
    currentdir = os.path.dirname(os.path.realpath(__file__))

    # Group composition: We will consider 2 skill levels: Good (q_good), and bad (q_bad) performers.
    # q_good and q2 are the standard deviation of the error distributions
    q_good = 1
    q_bad  = 5

    # Studied group composition
    model = sys.argv[1] # from [TEACHER, DIPLOMAT, CIRCLE] - LINE not implemented in this version
    resultsdir = model

    # Fix the model :
    if model == "TEACHER":
        flag = [True,True,True,True, True,False,False,False, True,False,False,False, True,False,False,False, True,False,False,False]
    elif model == "DIPLOMAT":
        flag = [True,True,True,True, True,True,False,False, True,True,False,False, True,False,False,True, True,False,False,True]
    elif model == "CIRCLE":
        flag = [True,False,False,True, True,True,False,False, False,True,True,False, False,False,True,True, True,False,False,True]
    elif model == "LINE":
        flag = [True,False,False,False, True,True,False,False, False,True,True,False, False,False,True,True, False,False,False,True]
    else:
        print("Model not supported. Please choose in [TEACHER, DIPLOMAT, CIRCLE, LINE].")
        sys.exit(-1)

    # We will vary the weights in the influence network across the following values:
    #WEIGHTS = np.arange(0,1.2,0.2) # Note : this excludes 1.2
    WEIGHTS = [0.0, 0.33, 0.66, 1.0]
    #WEIGHTS = [0.33, 0.66]

    # All 20 weight values of the network
    ##############
    w12 = WEIGHTS if flag[0] else [0]
    w13 = WEIGHTS if flag[1] else [0]
    w14 = WEIGHTS if flag[2] else [0]
    w15 = WEIGHTS if flag[3] else [0]
    ##############
    w21 = WEIGHTS if flag[4] else [0]
    w23 = WEIGHTS if flag[5] else [0]
    w24 = WEIGHTS if flag[6] else [0]
    w25 = WEIGHTS if flag[7] else [0]
    ##############
    w31 = WEIGHTS if flag[8] else [0]
    w32 = WEIGHTS if flag[9] else [0]
    w34 = WEIGHTS if flag[10] else [0]
    w35 = WEIGHTS if flag[11] else [0]
    ##############
    w41 = WEIGHTS if flag[12] else [0]
    w42 = WEIGHTS if flag[13] else [0]
    w43 = WEIGHTS if flag[14] else [0]
    w45 = WEIGHTS if flag[15] else [0]
    ##############
    w51 = WEIGHTS if flag[16] else [0]
    w52 = WEIGHTS if flag[17] else [0]
    w53 = WEIGHTS if flag[18] else [0]
    w54 = WEIGHTS if flag[19] else [0]
    ##############

    filenames = []
    if model == "CIRCLE" or model == "DIPLOMAT":
        for i in range(10):
            filenames += [os.path.join(model,model+str(i)+'_out.npy')]
    else:
        filenames += [os.path.join(model,model+'.npy')]

    data = np.load(filenames[0], allow_pickle=True).item()
    allErrorCubes = data['allErrorCubes']

    if model == "CIRCLE" or model == "DIPLOMAT":
        for f in filenames[1:]:
            tmp = np.load(f, allow_pickle=True).item()
            allErrorCubes = np.concatenate((allErrorCubes,tmp['allErrorCubes']),axis=1)

    test_experiment = create_experiment(model, data['SKILLS'], project_path=resultsdir)
    test_experiment.flag = flag
    test_experiment.N = data['N']
    test_experiment.nbReps = data['nbReps']
    test_experiment.nbSpeakingTurns = data['nbSpeakingTurns']
    test_experiment.nbRounds = data['nbRounds'] # 30
    test_experiment.initial_WEIGHT_Scalar_SocialLearning = 0.5

    test_experiment.opt_grid = np.array(list(itertools.product(w12,w13,w14,w15,w21,w23,w24,w25,w31,w32,w34,w35,w41,w42,w43,w45,w51,w52,w53,w54))).reshape(-1,20)

    test_experiment.allErrorCubes = allErrorCubes
    best_weight_configurations = test_experiment.determine_best_weight_configuration(True, 50)
    simulated_weight_values, simulated_error_values = test_experiment.weight_learning_over_discussion_rounds()
    test_experiment.analyze_social_discounting_effect(simulated_error_values, best_weight_configurations, 30)
    test_experiment.analyze_social_learning_pattern(simulated_weight_values, 2)


    print("Best weight configuration for ", model)
    for i in range(4):
        print("----------------------------")
        print(best_weight_configurations[i])
        print("----------------------------")

    print("Best weight configuration VARIANCE for ", model)
    for i in range(4):
        print("----------------------------")
        print(test_experiment.allBAMstd[i])
        print("----------------------------")
    
    ### Simulations ###
    impostor = False
    for rep in range(4):
        i = 0
        for skills in data['SKILLS']:
            W = best_weight_configurations[i]

            if impostor:
                W[0,1] = 0.0
                W[0,2] = 0.0
                W[0,3] = 0.0
                W[0,4] = 0.0

            i += 1
            for j in range(test_experiment.N):
                W[j, j] = 1.
            
            initial_estimate = np.multiply(np.random.randn(1,data['N']),skills).ravel()
            if impostor:
                initial_estimate[0] = -10
                namefile="impostor_skill"+str(i)+"_"+str(rep)+".jpg"
            else:
                namefile="skill"+str(i)+"_"+str(rep)+".jpg"

            test_experiment.simulate_and_plot_discussion(W, initial_estimate,namefile=namefile)


if __name__ == "__main__":
    main()