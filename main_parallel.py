import os, sys
import numpy as np
from experimental_design import create_experiment

def main():
    # Deal with directories
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)

    # Group composition: We will consider 2 skill levels: Good (q_good), and bad (q_bad) performers.
    # q_good and q2 are the standard deviation of the error distributions
    q_good = 1
    q_bad  = 5

    # Studied group composition
    model = sys.argv[1][0:-1] # from [TEACHER, DIPLOMAT, CIRCLE] - LINE not implemented in this version
    split = int(sys.argv[1][-1])
    print(model, ' nbr ', split)

    if model == "TEACHER":
        SKILLS = np.zeros((4,5))
        SKILLS[0,:] = [q_good, q_good, q_good, q_bad, q_bad]    # Good teacher - 2 good students 2 bad students
        SKILLS[1,:] = [q_bad, q_good, q_good, q_bad, q_bad]     # Bad teacher  - 2 good students 2 bad students
        SKILLS[2,:] = [q_good, q_good, q_good, q_good, q_good]  # Cross-compare : all good
        SKILLS[3,:] = [q_bad, q_bad, q_bad, q_bad, q_bad]       # Cross-compare : all bad
    elif model == "DIPLOMAT":
        SKILLS = np.zeros((4,5))
        SKILLS[0,:] = [q_good, q_good, q_good, q_bad, q_bad]    # Good diplomat - 1 bad group (left) 1 good group (right)
        SKILLS[1,:] = [q_bad, q_good, q_good, q_bad, q_bad]     # Bad diplomat  - 1 bad group (left) 1 good group (right)
        SKILLS[2,:] = [q_good, q_good, q_good, q_good, q_good]  # Cross-compare : all good
        SKILLS[3,:] = [q_bad, q_bad, q_bad, q_bad, q_bad]       # Cross-compare : all bad
    elif model == "CIRCLE":
        SKILLS = np.zeros((4,5))
        SKILLS[0,:] = [q_good, q_good, q_good, q_bad, q_bad]    # Three good people (neighbours)
        SKILLS[1,:] = [q_bad, q_good, q_good, q_bad, q_bad]     # Three bad people (neighbours)
        SKILLS[2,:] = [q_good, q_good, q_good, q_good, q_good]  # Cross-compare : all good
        SKILLS[3,:] = [q_bad, q_bad, q_bad, q_bad, q_bad]       # Cross-compare : all bad
        # define class that organizes how experiments are run - via this class every model configurations can be run
    # elif model == "LINE":
    #     nSkillset = 3
    #     SKILLS = np.zeros((nSkillset,5))
    #     SKILLS[0,:] = [q2, q2, q2, q2, q2] # Everyone average
    #     SKILLS[1,:] = [q_bad, q_bad, q2, q_good, q_good] # Bad head - Good tail
    #     SKILLS[2,:] = [q_good, q_good, q2, q_bad, q_bad] # Good head - Bad tail
    else:
        print("Unknown model :(")
        sys.exit(-1)
        
    # initialize experiment
    test_experiment = create_experiment(model, SKILLS, project_path=currentdir, split=split)
    test_experiment.conduct_experiment()

if __name__ == "__main__":
    main()