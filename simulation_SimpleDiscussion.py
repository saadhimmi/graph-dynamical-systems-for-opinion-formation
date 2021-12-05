import numpy as np

def simDiscussionFast( N , W , skills, nbSpeakingTurns, allSpeakers):
    '''
    
    This function simulates a simple group discussion, with the following
    parameters:
    N                 : The group size
    W                 : The weight matrix (NxN). 
    skills            : The skills of the N members (1xN)
    nbSpeakingTurns   : The duration of the discussion
    allSpeakers       : The speaking order
    
    '''
    
    # Initial estimates of the group members
    est = np.multiply(np.random.randn(1,N),skills)
    iEst = est 
    
    # Data structure
    allErr = np.zeros((nbSpeakingTurns+1 , N))
    allErr[:] = np.NaN
    allErr[0,:] = np.abs(iEst)
    
    # Discussion starts
    for t in range(nbSpeakingTurns):
        # choose a speaker at random
        speaker = allSpeakers[t]
        
        speakerEst = est[0,speaker]
        
        # influence of the speaker on the estimate of the others
        for i in range(N):
            # weights assigned to that speaker
            w = W[i,speaker]
            
            # revised estimate
            newEst = est[0,i] + w*(speakerEst - est[0,i])
            est[0,i] = newEst
            
        allErr[t+1,:] = np.abs(est)
        
    fEst = est
    
    return iEst, fEst, allErr

def simDiscussionFast_simultaneous(N, W, skills, nbSpeakingTurns, allSpeakers, nbExp):
    '''

    This function simulates a simple group discussion, with the following
    parameters:
    N                 : The group size
    W                 : The weight matrix (nbExp x N x N).
    skills            : The skills of the N members (1xN)
    nbSpeakingTurns   : The duration of the discussion
    allSpeakers       : The speaking order (nbExp x nbSpeakingTurns)
    nbExp             : Number of repetitions

    '''
    # Initial estimates of the group members
    est = np.multiply(np.random.randn(nbExp, N), skills) # nbExp x N
    iEst = est

    # Data structure
    allErr = np.zeros((nbExp, nbSpeakingTurns + 1, N)) # nbExp x nbSpeakingTurns+1 x N
    allErr[:] = np.NaN
    allErr[:, 0, :] = np.abs(iEst)

    # Discussion starts
    for t in range(nbSpeakingTurns):
        # choose a speaker at random
        speaker_t = allSpeakers[:,t].tolist()
        speakerEst = np.diag(est[:, speaker_t]).reshape(-1,1) # nbExp x 1
        

        # influence of the speaker on the estimate of the others
        for i in range(N):
            # weights assigned to that speaker
            #w = W[:, i, speaker_t].reshape(-1,1) # nbExp x 1
            w = np.diag(W[:, i, speaker_t].reshape(nbExp,nbExp)).reshape(-1,1)

            # revised estimate
            newEst = est[:, i].reshape(-1,1) + w * (speakerEst - est[:, i].reshape(-1,1))
            est[:, i:i+1] = newEst

        allErr[:, t + 1, :] = np.abs(est) # nbExp x 1

    fEst = est

    return iEst, fEst, allErr


def simDiscussionAndUpdateWeights( N , W , skills, bias, nbSpeakingTurns, allSpeakers, nbExp,
                                   ALLW, ALLE, idS, idB, speaking_round, learning_sensitivity):
    '''
    
    This function simulates a simple group discussion, with the following
    parameters:
    N                 : The group size
    W                 : The weight matrix (nbExp x N x N). 
    skills            : The skills of the N members (1xN)
    bias              : The bias evaluated
    nbSpeakingTurns   : The duration of the discussion
    allSpeakers       : The speaking order (nbExp x  nbSpeakingTurns)
    nbExp             : Number of repetitions
    ALLW              : influence weights of each pair of individuals over the speaking rounds (nbSkills x nbBias x nbReps x N x N x nbRounds)
    ALLE              : error over the speaking rounds (nbSkills x nbBias x nbReps x 1 x nbRounds)
    idS               : current skill index
    idB               : current bias index
    speaking_round    : current speaking round index
    '''
    # initiate array to save the Last discussion round´s relative error
    DE = np.zeros((nbExp, N, N))
    _, fEst, allErr = simDiscussionFast_simultaneous(N,W,skills,nbSpeakingTurns,allSpeakers,nbExp)

    # Computation of the matrix DE storing the relative error of all pairs of individuals
    for n in range(nbExp):
        for i in range(N):
            # Initial estimate of the focus individual
            err0 = allErr[n, 0, i] # nbExp x nbSpeakingTurns+1 x N

            # loop through all other individuals
            for j in range(N):
                if j != i:
                    # first expressed estimate of the partner j
                    k = np.argwhere(allSpeakers[n,:] == j)

                    if not np.all(k == 0):
                        firstTime = k[0]
                    else:
                        firstTime = np.NaN

                    if not np.isnan(firstTime):
                        # Corresponding expressed estimate
                        err = allErr[n, firstTime + 1, j]
                        DE[n, i, j] = (err0 - err)

        W[DE > (bias)] = W[DE > (bias)] + learning_sensitivity[DE > (bias)]
        W[DE < (bias)] = W[DE < (bias)] - learning_sensitivity[DE < (bias)]
        W[W > 1] = 1
        W[W < 0] = 0

        for i in range(nbExp):
            for j in range(N):
                W[i, j, j] = 1

        ALLW[idS, idB, :,:, :, speaking_round] = W
        ALLE[idS, idB, :, 0, speaking_round] = np.mean(np.abs(fEst))






    
