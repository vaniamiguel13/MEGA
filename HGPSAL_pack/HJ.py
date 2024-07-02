import numpy as np
import time

def GetOption(option, options, default_opt):
    if options is None or not isinstance(options, dict):
        return default_opt[option]
    return options.get(option, default_opt[option])

def ObjEval(Problem, x, *args):
    try:
        ObjValue = Problem['ObjFunction'](x, *args)
        if 'Stats' in Problem:
            Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise RuntimeError(f"User supplied objective function failed with the following error:\n{str(e)}")
    return ObjValue, Problem

def Projection(Problem, x):
    return np.clip(x, Problem['LB'], Problem['UB'])

def Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *args):
    if rho > 0:
        min_fx, Problem = ObjEval(Problem, x + s, *args)
        rho = fx - min_fx
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_fx, rho, *args)
    if rho <= 0:
        s = np.zeros_like(s)
        rho = 0
        min_fx = fx
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_fx, rho, *args)
    return s, Problem

def Coordinate_Search(Problem, s, delta, e, x, min_fx, rho, *args):
    for i in range(Problem['Variables']):
        s1 = s + delta * e[:, i]
        x1 = Projection(Problem, x + s1)
        fx1, Problem = ObjEval(Problem, x1, *args)
        if fx1 < min_fx:
            rho = min_fx - fx1
            min_fx = fx1
            s = s1
        else:
            s1 = s - delta * e[:, i]
            x1 = Projection(Problem, x + s1)
            fx1, Problem = ObjEval(Problem, x1, *args)
            if fx1 < min_fx:
                rho = min_fx - fx1
                min_fx = fx1
                s = s1
    return s, rho, Problem

def HJ(Problem, x0, delta=None, Options=None, *args):
    DefaultOpt = {'MaxObj': 2000, 'MaxIter': 200, 'DeltaTol': 1e-6, 'Theta': 0.5}

    if isinstance(Problem, str) and Problem.lower() == 'defaults':
        return DefaultOpt

    if delta is None:
        delta = np.ones_like(x0)
    if Options is None:
        Options = {}

    MaxEval = GetOption('MaxObj', Options, DefaultOpt)
    MaxIt = GetOption('MaxIter', Options, DefaultOpt)
    DelTol = GetOption('DeltaTol', Options, DefaultOpt)
    theta = GetOption('Theta', Options, DefaultOpt)

    Problem['Stats'] = {'Algorithm': 'Hooke and Jeeves', 'Iterations': 0, 'ObjFunCounter': 0, 'Message': '', 'Time': 0}

    x = Projection(Problem, x0)
    fx, Problem = ObjEval(Problem, x, *args)

    e = np.eye(Problem['Variables'])
    s = np.zeros_like(x)
    rho = 0

    start_time = time.time()

    while np.linalg.norm(delta) > DelTol and Problem['Stats']['ObjFunCounter'] < MaxEval and Problem['Stats'][
        'Iterations'] < MaxIt:
        s, Problem = Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *args)
        x_trial = Projection(Problem, x + s)
        fx1, Problem = ObjEval(Problem, x_trial, *args)
        rho = fx - fx1
        if rho > 0:
            x = x_trial
            fx = fx1
        else:
            delta *= theta
        Problem['Stats']['Iterations'] += 1

    if Problem['Stats']['Iterations'] >= MaxIt:
        Problem['Stats']['Message'] = 'HJ: Maximum number of iterations reached'
    elif Problem['Stats']['ObjFunCounter'] >= MaxEval:
        Problem['Stats']['Message'] = 'HJ: Maximum number of objective function evaluations reached'
    elif np.linalg.norm(delta) <= DelTol:
        Problem['Stats']['Message'] = 'HJ: Stopping due to step size norm inferior to tolerance'
    print(Problem['Stats']['Message'])

    Problem['Stats']['Time'] = round(time.time() - start_time, 2)
    RunData = Problem['Stats']
    return x, fx, RunData



# # Example usage
# if __name__ == "__main__":
#     def sphere_function(x):
#         return np.sum(x ** 2)
#
#
#     Problem = {
#         'Variables': 2,
#         'ObjFunction': sphere_function,
#         'LB': np.array([-15, -15]),
#         'UB': np.array([15, 15])
#     }
#     x0 = np.array([4, 4])
#     delta = np.array([0.1, 0.1])
#     Options = {
#         'MaxObj': 50000,
#         'MaxIter': 5000,
#         'DeltaTol': 1e-6,
#         'Theta': 0.5
#     }
#     x, fx, RunData = HJ(Problem, x0, delta, Options)
#     print(f"Best Solution: {x}")
#     print(f"Objective Value: {fx}")
#     print(f"Run Data: {RunData}")
