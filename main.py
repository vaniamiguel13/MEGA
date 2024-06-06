import numpy as np
import matplotlib.pyplot as plt

from MEGACon.MEGA_4 import MEGAcon
from cost_function import cost_function
from constraints import constraints
from parameters import LB, UB
from variables import initial_values


def main():
    Problem = {
        'Variables': len(LB),
        'ObjFunction': cost_function,
        'Objectives': 2,
        'LB': np.array(LB),
        'UB': np.array(UB),
        'Constraints': constraints,
        'c_dim': 1,
        'ceq_dim': 99
    }

    Options = {
        'Verbosity': 2,
        'PopSize': 15,
        'MaxObj': 2000,
        'MaxGen': 1000,
        'CTol': 1e-1,
        'CeqTol': 1e3,
        'CAdaptFlag': 0,
    }

    initial_population = [np.array(initial_values)]

    NonDomPoint, FrontPoint, RunData = MEGAcon(Problem, initial_population, Options)

    print("Nondominated Points (Decision Variables):")
    print(NonDomPoint)
    print("\nObjective Values of Nondominated Points:")
    print(FrontPoint['f'])
    print("\nConstraint Values (c) of Nondominated Points:")
    print(FrontPoint['c'])
    print("\nEquality Constraint Values (ceq) of Nondominated Points:")
    print(FrontPoint['ceq'])
    print("\nRun Data:")
    print(RunData)

    plt.figure()
    plt.plot(FrontPoint['f'][:, 0], FrontPoint['f'][:, 1], 'x')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
