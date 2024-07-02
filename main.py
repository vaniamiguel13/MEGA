import numpy as np
from MEGACon.MEGA_C2 import mega_con
from HGPSAL_pack.HGPSAL import HGPSAL
from cost_function import cost_function
from constraints import constraints
from parameters import LB, UB
from variables import initial_values

def main():
    Problem = {
        'Variables': len(LB),
        'ObjFunction': cost_function,
        'LB': np.array(LB),
        'UB': np.array(UB),
        'Constraints': constraints,
        'x0': initial_values
    }
    Options = {'verbose': 1, 'maxit': 1000000, 'maxet': 2000000, 'max_objfun': 2000000


    }

    x, fx, c, ceq, la, stats = HGPSAL(Problem, Options)
    print(f"Best Solution: {x}")
    print(f"Objective Value: {fx}")
    print(f"Constraint Values: {c}")
    print(f"Equality Constraint Values: {ceq}")
    print(f"Augmented Lagrangian: {la}")
    print(f"Run Data: {stats}")




if __name__ == '__main__':
    main()


