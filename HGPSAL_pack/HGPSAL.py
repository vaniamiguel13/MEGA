import numpy as np
from HGPSAL_pack.penalty import *
from HGPSAL_pack.HJ import HJ
from HGPSAL_pack.rGA import rGA
import time


def GetOption(option, options, default_opt):
    if options is None or not isinstance(options, dict):
        return default_opt[option]
    return options.get(option, default_opt[option])


def Projection(Problem, x):
    return np.clip(x, Problem['LB'], Problem['UB'])


def mega_(lambda_, ldelta, miu, teta_tol):
    return 1 / max(1, (1 + np.linalg.norm(lambda_) + np.linalg.norm(ldelta) + (1 / miu)) / teta_tol)


def lag(x, Problem, alg):
    Value = penalty2(Problem, x, alg)
    return Value['la']


def HGPSAL(Problem, options=None, *args):
    DefaultOpt = {
        'lambda_min': -1e12, 'lambda_max': 1e12, 'teta_tol': 1e12, 'miu_min': 1e-12, 'miu0': 1, 'csi': 0.5, 'eta0': 1,
        'omega0': 1, 'alfaw': 0.9, 'alfa_eta': 0.9 * 0.9, 'betaw': 0.9, 'beta_eta': 0.5 * 0.9, 'gama1': 0.5,
        'teta_miu': 0.5,
        'pop_size': 40, 'elite_prop': 0.1, 'tour_size': 2, 'pcross': 0.9, 'icross': 20, 'pmut': 0.1, 'imut': 20,
        'gama': 1, 'delta': 1, 'teta': 0.5, 'eta_asterisco': 1.0e-2, 'epsilon_asterisco': 1.0e-6,
        'cp_ga_test': 0.1, 'cp_ga_tol': 1.0e-6, 'delta_tol': 1e-6, 'maxit': 100, 'maxet': 200, 'max_objfun': 20000,
        'verbose': 0
    }

    if options == None:
        options = {}

    if Problem is None:
        raise ValueError('Invalid number of arguments.')

    if 'Variables' not in Problem or not Problem['Variables']:
        raise ValueError('Problem dimension is missing.')

    x0 = Problem.get('x0', [])

    if 'LB' not in Problem:
        raise ValueError('Problem lower bounds are missing.')
    lb = Problem['LB']

    if 'UB' not in Problem:
        raise ValueError('Problem upper bounds are missing.')
    ub = Problem['UB']

    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('Objective function name is missing.')

    if 'Constraints' not in Problem or not Problem['Constraints']:
        raise ValueError('Function constraints are missing.')

    if 'Variables' not in Problem or not Problem['Variables']:
        raise ValueError('Problem dimension is missing.')

    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('Objective function name is missing.')

    if 'Constraints' not in Problem:
        raise ValueError('Function constraints are missing.')

    if 'pop_size' not in options or options['pop_size'] < Problem['Variables']:
        options['pop_size'] = min(20 * Problem['Variables'], 200)
        print('HGPSAL: rGA population size set to', options['pop_size'])

    if 'pmut' in options:
        print('HGPSAL: rGA mutation probability set to', options['pmut'])
    else:
        options['pmut'] = 1 / Problem['Variables']
        print('HGPSAL: rGA mutation probability set to', options['pmut'])

    opt = DefaultOpt.copy()
    if options:
        for key in DefaultOpt:
            opt[key] = options.get(key, DefaultOpt[key])

    if len(x0) == 0:
        x = np.random.uniform(low=lb, high=ub, size=Problem['Variables'])
    else:
        x = np.array(x0)

    x = Projection(Problem, x)

    if 'Stats' not in Problem:
        Problem['Stats'] = {'ObjFunCounter': 0, 'GenCounter': 0, 'Best': [], 'Worst': [], 'Mean': [], 'Std': []}

    try:
        fx, Problem = ObjEval(Problem, x, *args)
    except Exception as e:
        raise RuntimeError(
            f"Cannot continue because user supplied objective function failed with the following error:\n{str(e)}")

    try:
        c, ceq = ConsEval(Problem, x, *args)
    except Exception as e:
        raise RuntimeError(
            f"Cannot continue because user supplied function constraints failed with the following error:\n{str(e)}")
    Problem['m'] = len(ceq)
    Problem['p'] = len(c)


    alg = {
        'lambda': np.ones(Problem['m']) if Problem['m'] > 0 else np.array([]),
        'ldelta': np.ones(Problem['p']) if Problem['p'] > 0 else np.array([]),
        'miu': opt['miu0'],
        'alfa': min(opt['miu0'], opt['gama1']),
        'omega': opt['omega0'] * (min(opt['miu0'], opt['gama1']) ** opt['alfaw']),
        'epsilon': opt['omega0'] * (min(opt['miu0'], opt['gama1']) ** opt['alfaw']) * mega_(np.ones(Problem['m']),
                                                                                            np.ones(Problem['p']),
                                                                                            opt['miu0'],
                                                                                            opt['teta_tol']),
        'eta': opt['eta0'] * (min(opt['miu0'], opt['gama1']) ** opt['alfa_eta']),
        'delta': np.ones(Problem['Variables']) if opt['delta'] == 1 else x0 * opt['gama']
    }

    stats = {
        'extit': 0, 'objfun': 0, 'x': [x], 'fx': [fx], 'c': [c], 'ceq': [ceq],
        'history': [['Iter', 'fx rGA', 'nf rGA', 'fx HJ', 'nf HJ']]
    }

    global_search = True

    tic = time.time()

    while stats['extit'] <= opt['maxet'] and stats['objfun'] <= opt['max_objfun']:
        stats['extit'] += 1
        stats['history'].append([stats['extit']])

        Probl = Problem.copy()
        Probl['ObjFunction'] = lambda x, *args: lag(x, Problem, alg)

        if global_search:
            InitialPopulation = [{'x': x}]
            Options = {
                'PopSize': opt['pop_size'], 'EliteProp': opt['elite_prop'], 'TourSize': opt['tour_size'],
                'Pcross': opt['pcross'], 'Icross': opt['icross'], 'Pmut': opt['pmut'], 'Imut': opt['imut'],
                'CPTolerance': alg['epsilon'], 'CPGenTest': opt['cp_ga_test'], 'MaxGen': opt['maxit'],
                'MaxObj': opt['max_objfun'], 'Verbosity': opt['verbose']
            }
            x, fval, RunData = rGA(Probl, InitialPopulation, Options, Problem, alg)
            stats['objfun'] += RunData['ObjFunCounter']
            stats['history'][stats['extit']].extend([fval, RunData['ObjFunCounter']])

            if opt['verbose']:
                print(f'GA external it: {stats["extit"]}')
                print(x, fval, RunData)

        Options = {
            'MaxIter': opt['maxit'], 'MaxObj': opt['max_objfun'], 'DeltaTol': alg['epsilon'], 'Theta': opt['teta']
        }
        x, fval, Rundata = HJ(Probl, x, alg['delta'], Options, Problem, alg)
        stats['objfun'] += Rundata['ObjFunCounter']
        stats['history'][stats['extit']].extend([fval, Rundata['ObjFunCounter']])

        if opt['verbose']:
            print(f'HJ external it: {stats["extit"]}')
            print(x, fval, Rundata)

        Value = penalty2(Problem, x, alg)
        c = np.array(Value['c'])
        ceq = np.array(Value['ceq'])
        fx = Value['fx']
        la = Value['la']

        stats['x'].append(x)
        stats['fx'].append(fx)
        stats['c'].append(c)
        stats['ceq'].append(ceq)

        if opt['verbose']:
            print(f'Iteration {stats["extit"]}')
            print(f'Solution: {x}')
            print(f'Objective Value: {fx}')
            print(f'Inequality Constraints: {c}')
            print(f'Equality Constraints: {ceq}')
            print(f'Augmented Lagrangian: {la}')

        if (not alg['lambda'].size and not alg['ldelta'].size):
            break

        max_i = max(abs(ceq)) if len(ceq) > 0 else 0
        v = max(max(c), max(alg['ldelta'] * np.abs(c))) if len(c) > 0 else 0

        norma_lambda = np.linalg.norm(alg['lambda'])
        norma_x = np.linalg.norm(x)

        alg['ldelta'] = np.clip(np.maximum(0, alg['ldelta'] + c / alg['miu']), opt['lambda_min'], opt['lambda_max'])

        if (max_i <= alg['eta'] * (1 + norma_x) and v <= alg['eta'] * (1 + norma_lambda)):
            if (alg['epsilon'] < opt['epsilon_asterisco'] and max_i <= opt['eta_asterisco'] * (1 + norma_x) and v <=
                    opt['eta_asterisco'] * (1 + norma_lambda) and global_search == 0):
                stats['message'] = 'HGPSAL_pack: Tolerance of constraints violations satisfied.'
                print(stats['message'])
                break
            else:
                alg['lambda'] = np.clip(alg['lambda'] + ceq / alg['miu'], opt['lambda_min'], opt['lambda_max'])
                alg['alfa'] = min(alg['miu'], opt['gama1'])
                alg['omega'] *= (alg['alfa'] ** opt['betaw'])
                alg['epsilon'] = alg['omega'] * mega_(alg['lambda'], alg['ldelta'], alg['miu'], opt['teta_tol'])
                alg['eta'] *= (alg['alfa'] ** opt['beta_eta'])
        else:
            alg['miu'] = max(min(alg['miu'] * opt['csi'], alg['miu'] ** opt['teta_miu']), opt['miu_min'])
            alg['alfa'] = min(alg['miu'], opt['gama1'])
            alg['omega'] = opt['omega0'] * (alg['alfa'] ** opt['alfaw'])
            alg['epsilon'] = alg['omega'] * mega_(alg['lambda'], alg['ldelta'], alg['miu'], opt['teta_tol'])
            alg['eta'] = opt['eta0'] * (alg['alfa'] ** opt['alfa_eta'])
            global_search = 1

    if stats['extit'] > opt['maxet']:
        stats['message'] = 'HGPSAL: Maximum number of external iterations reached.'
        print(stats['message'])
    if stats['objfun'] > opt['max_objfun']:
        stats['message'] = 'HGPSAL: Maximum number objective function evaluations reached.'
        print(stats['message'])

    stats['Time'] = round(time.time() - tic, 2)

    return x, fx, c, ceq, la, stats


#if __name__ == "__main__":
    # def sphere_function(x):
    #     return np.sum(x ** 2)
    #
    # def constraints(x):
    #     c = [x[0] + x[1] - 1]  # inequality constraint
    #     ceq = [x[0]*2 + 3]  # equality constraint
    #     return c, ceq
    #
    # Problem = {
    #     'Variables': 2,
    #     'ObjFunction': sphere_function,
    #     'Constraints': constraints,
    #     'LB': np.array([-15, -15]),
    #     'UB': np.array([15, 15]),
    #     'x0': np.array([2,3 ])
    # }
    # options = {
    #     'maxit': 50000,
    #     'maxet': 10000,
    #     'verbose': 1
    # }
    # x, fx, c, ceq, la, stats = HGPSAL(Problem, options)
    # print(f"Best Solution: {x}")
    # print(f"Objective Value: {fx}")
    # print(f"Constraint Values: {c}")
    # print(f"Equality Constraint Values: {ceq}")
    # print(f"Augmented Lagrangian: {la}")
    # print(f"Run Data: {stats}")

# def rosenbrock_function(x):
#     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
#
# def constraints(x):
#     c = [x[0]**2 + x[1]**2 - 1]  # inequality constraint
#     ceq = [x[0] + x[1] - 1]  # equality constraint
#     return c, ceq

# if __name__ == "__main__":
#     Problem = {
#         'Variables': 2,
#         'ObjFunction': rosenbrock_function,
#         'Constraints': constraints,
#         'LB': np.array([-2, -2]),
#         'UB': np.array([2, 2]),
#         'x0': np.array([0.5, 0.5])
#     }
#     options = {
#         'maxit': 50000,
#         'maxet': 10000,
#         'verbose': 1
#     }
#     # Assuming HGPSAL is defined somewhere in your code base
#     x, fx, c, ceq, la, stats = HGPSAL(Problem, options)
#     print(f"Best Solution: {x}")
#     print(f"Objective Value: {fx}")
#     print(f"Constraint Values: {c}")
#     print(f"Equality Constraint Values: {ceq}")
#     print(f"Augmented Lagrangian: {la}")
#     print(f"Run Data: {stats}")


# def himmelblau_function(x):
#     return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
#
#
# def constraints(x):
#     c = [x[0] ** 2 + x[1] ** 2 - 5]  # inequality constraint
#     ceq = [x[0] - x[1]]  # equality constraint
#     return c, ceq
#
#
# if __name__ == "__main__":
#     Problem = {
#         'Variables': 2,
#         'ObjFunction': himmelblau_function,
#         'Constraints': constraints,
#         'LB': np.array([-5, -5]),
#         'UB': np.array([5, 5]),
#         'x0': np.array([2.0, 3.0])
#     }
#     Options = {
#         'verbose': 1
#     }
#
#     x, fx, c, ceq, la, stats = HGPSAL(Problem, options=Options)
#     print(f"Best Solution: {x}")
#     print(f"Objective Value: {fx}")
#     print(f"Constraint Values: {c}")
#     print(f"Equality Constraint Values: {ceq}")
#     print(f"Augmented Lagrangian: {la}")
#     print(f"Run Data: {stats}")
