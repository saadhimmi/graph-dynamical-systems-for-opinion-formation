"""
Description
"""

# load necessary modules for running this script
import os, sys
import itertools
import numpy as np
import time
from matplotlib import pyplot as plt

# load custom modules necessary for running this script
from simulation_SimpleDiscussion import simDiscussionFast_simultaneous, simDiscussionAndUpdateWeights

    
class create_experiment():
    def __init__(self, model,  # Group composition (links)
                       SKILLS, # Group composition: Skill levels of the participants, e.g. We will consider 2 skill levels: Good (q_good), and bad (q2) performers.
                       project_path,
                       split = None,): # Makes distributed computing easier

        self.project_path = project_path
        self.model = model
        self.split = split
        if split is not None:    
            self.parallel = True
            self.output_filename = os.path.join(project_path,model+str(split)+"_out.npy")
        else:
            self.parallel = False
            self.output_filename = os.path.join(project_path,model+"_out.npy")

        self.SKILLS = SKILLS
        self.data = {'model':self.model, 'SKILLS':self.SKILLS, 'split':self.split}
        self.show_figures = False

    def conduct_experiment(self,
                           FLAG_optimal_weight_configuration=True,
                           FLAG_plot_optimal_weights=True,
                           FLAG_social_learning=True,
                           N=5,# Group size
                           nbReps = 50, # Number of replications of the experiment
                           nbSpeakingTurns=15, # Number of speaking rounds, called Nr in the paper
                           nbRounds = 30,# Number of learning / discussion rounds, called NT in the paper
                           initial_WEIGHT_Scalar_SocialLearning=0.5,# initial weight Scalar of social learning experiment
                           xTop=50, # number of weight configurations that are used to calculate averaged optimal configuration
                           xLast=20,
                           xLast_learning_rounds=2, #number of last discussion rounds over which to calculate the average weights
    ):
        self.N = N
        self.nbReps = nbReps
        self.nbSpeakingTurns = nbSpeakingTurns
        self.nbRounds = nbRounds
        self.initial_WEIGHT_Scalar_SocialLearning = initial_WEIGHT_Scalar_SocialLearning # to be implemented

        self.data['N']= self.N
        self.data['nbReps']=self.nbReps
        self.data['nbSpeakingTurns']=self.nbSpeakingTurns
        self.data['nbRounds']=self.nbRounds

        if FLAG_optimal_weight_configuration:
            self.exhaustive_weight_configuration_search(self.model)
            best_weight_configurations = self.determine_best_weight_configuration(FLAG_plot_optimal_weights,xTop)
        if FLAG_social_learning:
            simulated_weight_values, simulated_error_values = self.weight_learning_over_discussion_rounds()
            self.analyze_social_discounting_effect(simulated_error_values, best_weight_configurations, xLast)
            self.analyze_social_learning_pattern(simulated_weight_values, xLast_learning_rounds)

        np.save(self.output_filename, self.data)

    def simulate_and_plot_discussion(self, W, iEst, namefile="default.jpg"):

        est = iEst 

        # Predefined random speaking turns for the examples
        allSpeakers = np.random.randint(self.N, size=self.nbSpeakingTurns)

        # Data structure for storing the results
        allEst = np.zeros((self.nbSpeakingTurns+1 , self.N))
        allEst[:] = np.NaN
        allEst[0,:] = est
        allErr = np.zeros((self.nbSpeakingTurns+1 , self.N))
        allErr[:] = np.NaN
        allErr[0,:] = np.abs(est)

        # Discussion starts here
        for t in range(self.nbSpeakingTurns):
            # Choose the next speaker
            speaker = int(allSpeakers[t])
            
            # Estimate of that speaker
            speakerEst = est[speaker]

            # influence of the speaker on the estimate of all the others
            for i in range(self.N):
                # Weight assigned to that speaker
                w = W[i,speaker]

                # Revised estimate
                newEst = est[i] + w*(speakerEst - est[i])
                est[i] = newEst
            # Store data
            allErr[t+1,:] = np.abs(est) 
            allEst[t+1 , :] = est
                            
        # End of the discussion
        # Display
        plt.figure(dpi=150)
        plt.plot(allEst[:,0] , '.-', linewidth=7.0)
        plt.plot(allEst[:,1:] , '.-')
        for t in range(self.nbSpeakingTurns):
            n = allSpeakers[t]
            plt.plot(t+1 , allEst[t+1 , n] , 'ko')
        plt.plot([-1, self.nbSpeakingTurns+1] , [0, 0] , 'k--')
        plt.title('Discussion Dynamics')
        plt.xlabel('Discussion Round')
        plt.ylabel('Judgement')
        plt.legend(['$p_1$', '$p_2$','$p_3$', '$p_4$', '$p_5$'])
        plt.xlim(-1,self.nbSpeakingTurns+1)
        plt.savefig(os.path.join(self.project_path, namefile))
        if self.show_figures:
            plt.show()


    # Exhaustive grid search over all possible weight configurations in one discussion round - N = 5
    def exhaustive_weight_configuration_search(self, model, print_time=True):
        #########
        #CODE HAS TO BE ADAPTED TO BE DYNAMIC AND CREATE MULTIPLE EXPERIMENTAL SCENARIOS
        ##########
        
        # Fix the model :
        if model == "TEACHER":
            self.flag = [True,True,True,True, True,False,False,False, True,False,False,False, True,False,False,False, True,False,False,False]
        elif model == "DIPLOMAT":
            self.flag = [True,True,True,True, True,True,False,False, True,True,False,False, True,False,False,True, True,False,False,True]
        elif model == "CIRCLE":
            self.flag = [True,False,False,True, True,True,False,False, False,True,True,False, False,False,True,True, True,False,False,True]
        elif model == "LINE":
            self.flag = [True,False,False,False, True,True,False,False, False,True,True,False, False,False,True,True, False,False,False,True]
        else:
            print("Model not supported. Please choose in [TEACHER, DIPLOMAT, CIRCLE, LINE].")
            sys.exit(-1)

        # We will vary the weights in the influence network across the following values:
        #self.WEIGHTS = np.arange(0,1.2,0.2) # Note : this excludes 1.2
        self.WEIGHTS = [0.0, 0.33, 0.66, 1.0]
        #self.WEIGHTS = [0.33, 0.66]

        # All 20 weight values of the network
        ##############
        w12 = self.WEIGHTS if self.flag[0] else [0]
        w13 = self.WEIGHTS if self.flag[1] else [0]
        w14 = self.WEIGHTS if self.flag[2] else [0]
        w15 = self.WEIGHTS if self.flag[3] else [0]
        ##############
        w21 = self.WEIGHTS if self.flag[4] else [0]
        w23 = self.WEIGHTS if self.flag[5] else [0]
        w24 = self.WEIGHTS if self.flag[6] else [0]
        w25 = self.WEIGHTS if self.flag[7] else [0]
        ##############
        w31 = self.WEIGHTS if self.flag[8] else [0]
        w32 = self.WEIGHTS if self.flag[9] else [0]
        w34 = self.WEIGHTS if self.flag[10] else [0]
        w35 = self.WEIGHTS if self.flag[11] else [0]
        ##############
        w41 = self.WEIGHTS if self.flag[12] else [0]
        w42 = self.WEIGHTS if self.flag[13] else [0]
        w43 = self.WEIGHTS if self.flag[14] else [0]
        w45 = self.WEIGHTS if self.flag[15] else [0]
        ##############
        w51 = self.WEIGHTS if self.flag[16] else [0]
        w52 = self.WEIGHTS if self.flag[17] else [0]
        w53 = self.WEIGHTS if self.flag[18] else [0]
        w54 = self.WEIGHTS if self.flag[19] else [0]
        ##############

        self.opt_grid = np.array(list(itertools.product(w12,w13,w14,w15,w21,w23,w24,w25,w31,w32,w34,w35,w41,w42,w43,w45,w51,w52,w53,w54))).reshape(-1,20)
        #self.opt_grid = np.array(np.meshgrid(w12,w13,w14,w15,w21,w23,w24,w25,w31,w32,w34,w35,w41,w42,w43,w45,w51,w52,w53,w54)).reshape(-1,20)
        print("TOTAL combinations to test : ", self.opt_grid.shape)
        
        if self.parallel:
            tmp_list_of_subarrays = np.array_split(self.opt_grid, 10, axis=0)
            self.opt_grid = tmp_list_of_subarrays[self.split]
            print("Current combinations to test : ", self.opt_grid.shape)

        allCubes = []
        # For all group compositions, we now search for the most efficient influence netowork
        t0 = time.time()
        for skillset in self.SKILLS:
            t = time.time()
            # Output variables
            MERR = np.zeros(self.opt_grid.shape[0])
            idx = 0
            MERR[:] = np.nan
            
            for _ in self.opt_grid:          
                # prepare the weight matrix
                W = np.zeros((self.nbReps,5,5))
                W[:] = np.NaN

                W[:,0,:] = [    1,    _[0],    _[1],   _[2],    _[3]]
                W[:,1,:] = [ _[4],       1,    _[5],   _[6],    _[7]]
                W[:,2,:] = [ _[8],    _[9],       1,  _[10],   _[11]]
                W[:,3,:] = [_[12],   _[13],   _[14],      1,   _[15]]
                W[:,4,:] = [_[16],   _[17],   _[18],  _[19],       1]
                
                # Prepare the speakers
                allSpeakers = np.random.randint(self.N, size=(self.nbReps, self.nbSpeakingTurns))
                
                # And run the simulations
                allFinalErrors = np.zeros((self.nbReps,self.N))
                allFinalErrors[:] = np.NaN

                _, fEst, _ = simDiscussionFast_simultaneous( self.N , W , skillset, self.nbSpeakingTurns , allSpeakers, self.nbReps)
                allFinalErrors = fEst

                # Store the average error of the group
                MERR[idx] = np.mean(np.abs(allFinalErrors[:])) 
                idx += 1

            allCubes.append(MERR)
            elapsed = time.time() - t
            print("Partial loop : {} sec.".format(elapsed))
        if print_time:    
            print('Final elapsed time : {} seconds'.format(time.time()-t0))

        self.allErrorCubes = np.array(allCubes)
        self.data['allErrorCubes'] = self.allErrorCubes

        if self.parallel:
            np.save(self.output_filename, self.data)
            sys.exit(-1)
    

    def determine_best_weight_configuration(self, # N = 5
                                            FLAG_plot_optimal_weights,
                                            xTop, # number of weight configurations that are used to calculate averaged optimal configuration
                                            ):
        # Find out the Best Configuration
        self.allBAM = np.zeros((np.size(self.SKILLS, 0),5,5))
        self.allBAMstd = np.zeros_like((self.allBAM))
        for idSkills in range(np.size(self.SKILLS, 0)):
            
            E = self.allErrorCubes[idSkills] # (size_opt_grid x 1) errors
            
            # Best xTop configs
            xTop = 50
            xs = np.argsort(E,axis=None) #(size_opt_grid x 1) ordered indices 
            
            bestConfig = self.opt_grid[xs[:xTop],:]
            
            bestAvgConfig = np.mean(bestConfig,axis=0) 
            bestAvgConfigstd = np.std(bestConfig,axis=0) 

            BAM = np.zeros((5,5)) 
            BAM[0,:] = [           np.NaN,    bestAvgConfig[0],    bestAvgConfig[1],     bestAvgConfig[2],    bestAvgConfig[3]]
            BAM[1,:] = [ bestAvgConfig[4],              np.NaN,    bestAvgConfig[5],     bestAvgConfig[6],    bestAvgConfig[7]]
            BAM[2,:] = [ bestAvgConfig[8],    bestAvgConfig[9],              np.NaN,    bestAvgConfig[10],   bestAvgConfig[11]]
            BAM[3,:] = [bestAvgConfig[12],   bestAvgConfig[13],   bestAvgConfig[14],               np.NaN,   bestAvgConfig[15]]
            BAM[4,:] = [bestAvgConfig[16],   bestAvgConfig[17],   bestAvgConfig[18],    bestAvgConfig[19],              np.NaN]
            
            BAMstd = np.zeros((5,5)) 
            BAMstd[0,:] = [              np.NaN,    bestAvgConfigstd[0],    bestAvgConfigstd[1],     bestAvgConfigstd[2],    bestAvgConfigstd[3]]
            BAMstd[1,:] = [ bestAvgConfigstd[4],                 np.NaN,    bestAvgConfigstd[5],     bestAvgConfigstd[6],    bestAvgConfigstd[7]]
            BAMstd[2,:] = [ bestAvgConfigstd[8],    bestAvgConfigstd[9],                 np.NaN,    bestAvgConfigstd[10],   bestAvgConfigstd[11]]
            BAMstd[3,:] = [bestAvgConfigstd[12],   bestAvgConfigstd[13],   bestAvgConfigstd[14],                  np.NaN,   bestAvgConfigstd[15]]
            BAMstd[4,:] = [bestAvgConfigstd[16],   bestAvgConfigstd[17],   bestAvgConfigstd[18],    bestAvgConfigstd[19],                 np.NaN]

                
            self.allBAM[idSkills] = BAM
            self.allBAMstd[idSkills] = BAMstd

        self.data['allBAM'] = self.allBAM
        self.data['allBAMstd'] = self.allBAMstd
            
        if FLAG_plot_optimal_weights:
            # Display
            plt.figure(figsize=(16, 5), dpi=150)

            for s in range(np.size(self.SKILLS, 0)):
                plt.subplot(1, 4, s + 1)
                plt.imshow(self.allBAM[s], extent=[0, 1, 0, 1])
                plt.clim(0, 1)
                plt.cm.get_cmap("jet")
                plt.title([str(self.SKILLS[s, :])])
                
            plt.suptitle('Best network configurations (weights wij) in 4 compositions')
            plt.tight_layout()
            plt.savefig(os.path.join(self.project_path,"best_weight_config.jpg"))
            if self.show_figures:
                plt.show()

        return  self.allBAM

    # define a function that enables to learn weights over the course of multiple discussion rounds
    def weight_learning_over_discussion_rounds(self,
                                                social_discounting_bias=(0,2.1,0.1), #list of values or tuple of values that is used for numpy arange command
                                                SKILLS=None,
                                                print_time=True,
                                                learning_val = 0.1):

        # initialize array of bias values over which we run the simulations
        if isinstance(social_discounting_bias,(int,float,list)):
            self.BIAS = np.array(social_discounting_bias,ndmin=2)
        elif isinstance(social_discounting_bias,tuple):
            self.BIAS = np.arange(*social_discounting_bias)

        if SKILLS is None:
            SKILLS = self.SKILLS
        elif isinstance(SKILLS, list):
            SKILLS  = np.array(SKILLS, ndmin=2)

        # initialize array to store the influence weights of each pair of individuals over the speaking rounds,
        # experimental conditions, the social discounting biases and the skill levels
        self.ALLW = np.zeros((np.size(SKILLS, 0), np.size(self.BIAS, 0), self.nbReps, self.N, self.N, self.nbRounds))
        # initialize ...
        self.ALLE = np.zeros((np.size(SKILLS, 0), np.size(self.BIAS, 0), self.nbReps, 1, self.nbRounds))

        # All 20 weight values of the network are the same 
        w0 = [self.initial_WEIGHT_Scalar_SocialLearning if bool else 0.0 for bool in self.flag]

        learning_sensitivity = np.zeros((self.nbReps,5,5))
        learning_sensitivity[:] = np.NaN
        ls = [learning_val if bool else 0.0 for bool in self.flag]
        learning_sensitivity[:,0,:] = [     1,  ls[0],   ls[1],  ls[2],   ls[3]]
        learning_sensitivity[:,1,:] = [ ls[4],      1,   ls[5],  ls[6],   ls[7]]
        learning_sensitivity[:,2,:] = [ ls[8],   ls[9],      1, ls[10],  ls[11]]
        learning_sensitivity[:,3,:] = [ls[12],  ls[13], ls[14],      1,  ls[15]]
        learning_sensitivity[:,4,:] = [ls[16],  ls[17], ls[18], ls[19],       1]

        t = time.time()
        for idS in range(np.size(SKILLS, 0)):
            skills = SKILLS[idS, :]
            for idB in range(len(self.BIAS)):
                bias = self.BIAS[idB]

                # initial weight matrix
                #W = np.ones((self.nbReps, self.N, self.N)) * self.initial_WEIGHT_Scalar_SocialLearning
                W = np.zeros((self.nbReps,5,5))
                W[:] = np.NaN
                W[:,0,:] = [     1,  w0[0],   w0[1],  w0[2],   w0[3]]
                W[:,1,:] = [ w0[4],      1,   w0[5],  w0[6],   w0[7]]
                W[:,2,:] = [ w0[8],   w0[9],      1, w0[10],  w0[11]]
                W[:,3,:] = [w0[12],  w0[13], w0[14],      1,  w0[15]]
                W[:,4,:] = [w0[16],  w0[17], w0[18], w0[19],       1]
                # set diagonal to 1
                for i in range(self.nbReps):
                    for j in range(self.N):
                        W[i, j, j] = 1.

                # Randomly initialize the speaking turns
                allSpeakers = np.random.randint(self.N, size=(self.nbReps, self.nbRounds, self.nbSpeakingTurns))

                # Loop through all discussion rounds
                for speaking_round in range(self.nbRounds):
                    speakers = allSpeakers[:, speaking_round, :]

                    # Simulate the discussion
                    simDiscussionAndUpdateWeights(self.N, W, skills, bias, self.nbSpeakingTurns, speakers, self.nbReps, 
                                                    self.ALLW, self.ALLE, idS, idB, speaking_round, learning_sensitivity)

        elapsed = time.time() - t
        if print_time:
            print('Elapsed time {} seconds'.format(elapsed))

        self.data['ALLW']=self.ALLW
        self.data['ALLE']=self.ALLE

        return self.ALLW, self.ALLE

    def analyze_social_discounting_effect(self,
                                          simulated_error_values, # nbSkills x nbBias x nbReps x 1 x nbRounds
                                          optimal_weight_configuration,
                                          xLast, # number of discussion rounds that are used to calculate averaged error
                                          ):

        # intialize array to save the average error of the group at the end of the number of discussion rounds
        # (i.e. after the internal structure of the group has emerged and stabilized)
        avgEndErr = np.zeros((np.size(self.SKILLS, 0), len(self.BIAS)))
        avgEndErr[:] = np.NaN
        E0 = np.zeros((1, np.size(self.SKILLS, 0)))
        E0[:] = np.NaN

        for idS in range(np.size(self.SKILLS, 0)):
            # retrieve the emerged Errors for a specific skill set
            OUTBias_E = simulated_error_values[idS]
            skills = self.SKILLS[idS, :]

            # % We do a new set of simulations with the optimal configurations.
            # The results will serve as a benchmark.
            W0 = np.ones((self.nbReps, self.N, self.N))
            for i in range(self.nbReps):
                W0[i,:,:] = optimal_weight_configuration[idS]
                for j in range(self.N):
                    W0[i, j, j] = 1

            # # % Correct a small mistake for idS=3 (goodperformer is at position 1).
            # if idS == 3:
            #     W0 = np.roll(W0, 2, 1)
            #     W0 = np.roll(W0, 2, 0)

            mErr0 = np.zeros((1, self.nbReps))
            mErr0[:] = np.NaN
            allSpeakers = np.random.randint(self.N, size=(self.nbReps, self.nbSpeakingTurns))

            _, fEst, _ = simDiscussionFast_simultaneous(self.N, W0, skills, self.nbSpeakingTurns, allSpeakers, self.nbReps)
            mErr0[0, :] = np.mean(np.abs(fEst), axis=1)

            E0[0, idS] = np.mean(mErr0)

            # Compute average error of the simulated groups averaged over the xLast rounds
            for idb in range(len(self.BIAS)):

                OUTE = OUTBias_E[idb]

                # Average errors over rounds
                temp = np.zeros((1, self.nbRounds))
                for r in range(self.nbReps):
                    temp = temp + (OUTE[r])
                temp = temp / self.nbReps

                avgEndErr[idS, idb] = np.mean(temp[-1 - xLast::])

        plt.figure(figsize=(16, 5), dpi=150)
        for i in range(np.size(E0, axis=1)):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.BIAS, avgEndErr[i, :], 'ro-')
            plt.plot([0, 3], [E0[0, i], E0[0, i]], 'b--')
            plt.title(str(self.SKILLS[i, :]))
            plt.xlabel('Bias')
            plt.xlim(0, 2)

        plt.suptitle('Impact of the social discounting bias')
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_path,"social_discounting_effect.jpg"))
        if self.show_figures:
            plt.show()

    # Compute the actual average structure that has emerged after the learning phase.
    def analyze_social_learning_pattern(self,
                                        simulated_weight_values,
                                        xLast_learning_rounds,  #number of last discussion rounds over which to calculate the average weights
                                        social_discounting_bias_values=[0,0.1,1]
                                        ):
        # initialize array for saving the ermerged weight values because of learning
        if social_discounting_bias_values is None:
            social_discounting_bias_values = [0., 3, 11]
        Wavg = np.zeros((np.size(self.SKILLS, 0), len(social_discounting_bias_values), self.nbRounds, 20))
        Wavg[:] = np.NaN
        Wstd = np.zeros_like(Wavg)
        Wstd[:] = np.NaN
        for idS in range(np.size(self.SKILLS, 0)):
            # define the indexes of the Bias values to retrieve the correct values from the simulation results
            bias_indices = [index for index in range(len(self.BIAS)) if self.BIAS[index] in social_discounting_bias_values]
            # loop over the defined social discounting bias values
            for i in range(len(social_discounting_bias_values)):
                # retrieve the index of the first bias value
                idB = bias_indices[i]
                # Get all possible weight values for all the discussion rounds
                for t in range(self.nbRounds):

                    linw = np.zeros((self.nbReps, 20))

                    for r in range(self.nbReps):
                        # retrieve the all weight configurations for the current skill configuration, bias value,
                        # experimental repitition and speaking round
                        w = simulated_weight_values[idS, idB, r, :, :, t]
                        linw[r, :] = [         w[0, 1], w[0, 2], w[0, 3], w[0, 4],
                                      w[1, 0],          w[1, 2], w[1, 3], w[1, 4],
                                      w[2, 0], w[2, 1],          w[2, 3], w[2, 4],
                                      w[3, 0], w[3, 1], w[3, 2],          w[3, 4],
                                      w[4, 0], w[4, 1], w[4, 2], w[4, 3]         ]

                    Wavg[idS, i, t, :] = np.mean(linw, axis=0)
                    Wstd[idS, i, t, :] = np.std(linw, axis=0)

        # Now display the average weights across the last xLast_learning_rounds  (final configuration)
        idd = 1

        plt.figure(figsize=(16, 10), dpi=150)
        for idS in range(np.size(self.SKILLS, 0)):

            # loop over the defined social discounting bias values
            for i in range(len(social_discounting_bias_values)):
                w = np.mean(Wavg[idS, i][Wavg.shape[2]-xLast_learning_rounds:, :], axis=0)

                plt.subplot(4, 3, idd)
                plt.stem(w)
                plt.ylim(0, 1)
                plt.xlabel('weight wij')
                plt.ylabel('value')
                plt.title(str(self.SKILLS[idS, :]))
                print(['Bias=', str(social_discounting_bias_values[i]), ' Composition:', str(self.SKILLS[idS, :]), '  weight wij: ', str(w)])
                idd += 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_path,"social_learning_pattern.jpg"))
        if self.show_figures:
            plt.show()