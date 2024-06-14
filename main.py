# import numpy as np
# import matplotlib.pyplot as plt
#
# from MEGACon.MEGA_C import mega_con
# from cost_function import cost_function
# from constraints import constraints
# from parameters import LB, UB
# from variables import initial_values

#
# def main():
#     Problem = {
#         'Variables': len(LB),
#         'ObjFunction': cost_function,
#         'Objectives': 2,
#         'LB': np.array(LB),
#         'UB': np.array(UB),
#         'Constraints': constraints,
#         'c_dim': 1,
#         'ceq_dim': 99
#     }
#
#     Options = {
#         'Verbosity': 2,
#         'PopSize': 150,
#         'MaxObj': 2000,
#         'MaxGen': 1000,
#         'CTol': 1e-1,
#         'CeqTol': 1e10,
#         'CAdaptFlag': 0,
#     }
#
#     initial_population = [np.array(initial_values)]
#
#     NonDomPoint, FrontPoint, RunData = mega_con(Problem, initial_population, Options)
#
#     print("Nondominated Points (Decision Variables):")
#     print(NonDomPoint)
#     print("\nObjective Values of Nondominated Points:")
#     print(FrontPoint['f'])
#     print("\nConstraint Values (c) of Nondominated Points:")
#     print(FrontPoint['c'])
#     print("\nEquality Constraint Values (ceq) of Nondominated Points:")
#     print(FrontPoint['ceq'])
#     print("\nRun Data:")
#     print(RunData)
#
#     plt.figure()
#     plt.plot(FrontPoint['f'][:, 0], FrontPoint['f'][:, 1], 'x')
#     plt.xlabel('f1')
#     plt.ylabel('f2')
#     plt.title('Pareto Front')
#     plt.grid(True)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

import numpy as np
import matplotlib.pyplot as plt

from MEGACon.MEGA_C2 import mega_con
from cost_function import cost_function
from constraints import constraints
from parameters import LB, UB
from variables import initial_values


import numpy as np
from MEGACon.MEGA_C2 import mega_con
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
        'PopSize': 150,
        'MaxObj': 2000,
        'MaxGen': 1000,
        'CTol': 1e-1,
        'CeqTol': 1e10,
        'CAdaptFlag': 0,
    }

    initial_population = [{'x': np.array(initial_values)}]

    results_df, run_data = mega_con(Problem, initial_population, Options)

    print(results_df)
    print("\nRun Data:")
    print(run_data)

    # Save the DataFrame to a CSV file
    results_df.to_csv('results.csv', index=False)
    print("\nResults saved to 'results.csv'")

if __name__ == '__main__':
    main()


# import numpy as np
# import matplotlib.pyplot as plt
#
# from MEGACon.MEGA_C import mega_con
#  # Ensure this import matches your project structure
#
# # Define a simple objective function with two objectives
# def cost_function(x):
#     return [np.sum(x**2), np.sum((x - 2)**2)]
#
# # Define simple constraints
# def constraints(x):
#     c = [np.sum(x) - 5]  # Inequality constraint
#     ceq = [np.prod(x) - 1]  # Equality constraint
#     return c, ceq
#
# # Define lower and upper bounds
# LB = np.zeros(5)  # 5 variables with lower bound of 0
# UB = np.ones(5) * 10  # 5 variables with upper bound of 10
#
# # Define initial values
# initial_values = np.ones(5)  # Initial values set to 1
#
# def main():
#     Problem = {
#         'Variables': len(LB),
#         'ObjFunction': cost_function,
#         'Objectives': 2,
#         'LB': np.array(LB),
#         'UB': np.array(UB),
#         'Constraints': constraints,
#         'c_dim': 1,
#         'ceq_dim': 1
#     }
#
#     Options = {
#         'Verbosity': 2,
#         'PopSize': 40,
#         'MaxObj': 2000,
#         'MaxGen': 1000,
#         'CTol': 1e-1,
#         'CeqTol': 1e-2,
#         'CAdaptFlag': 0,
#     }
#
#     initial_population = [{'x': np.array(initial_values)}]
#
#     NonDomPoint, FrontPoint, RunData = mega_con(Problem, initial_population, Options)
#
#     print("Nondominated Points (Decision Variables):")
#     print(NonDomPoint)
#     print("\nObjective Values of Nondominated Points:")
#     print(FrontPoint['f'])
#     print("\nConstraint Values (c) of Nondominated Points:")
#     print(FrontPoint['c'])
#     print("\nEquality Constraint Values (ceq) of Nondominated Points:")
#     print(FrontPoint['ceq'])
#     print("\nRun Data:")
#     print(RunData)
#
#     # Plot the Pareto front
#     plt.figure()
#     plt.scatter([fp[0] for fp in FrontPoint['f']], [fp[1] for fp in FrontPoint['f']], marker='x')
#     plt.xlabel('Objective 1')
#     plt.ylabel('Objective 2')
#     plt.title('Pareto Front')
#     plt.grid(True)
#     plt.show()
#
# if __name__ == '__main__':
#     main()
