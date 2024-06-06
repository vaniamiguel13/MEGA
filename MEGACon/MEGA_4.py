import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def MEGAcon(Problem, InitialPopulation, Options, *varargs):
    DefaultOpt = {
        'MaxObj': 2000,
        'MaxGen': 1000,
        'PopSize': 40,
        'Elite': 0.1,
        'TourSize': 2,
        'Pcross': 0.9,
        'Icross': 20,
        'Pmut': 0.1,
        'Imut': 20,
        'Sigma': 0.1,
        'CPTolerance': 1.0e-6,
        'CPGenTest': 0.01,
        'CTol': 1e-2,
        'CeqTol': 1e2,  # Set CeqTol to 1e2
        'NormType': np.inf,
        'NormCons': 1,
        'Verbosity': 0
    }

    def get_option(option, options, default_opt):
        return options.get(option, default_opt[option])

    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('Objective function name is missing.')
    if 'LB' not in Problem or 'UB' not in Problem or not isinstance(Problem['LB'], np.ndarray) or not isinstance(
            Problem['UB'], np.ndarray):
        raise ValueError('Population relies on finite bounds on all variables.')
    if len(Problem['LB']) != len(Problem['UB']):
        raise ValueError('Lower bound and upper bound arrays length mismatch.')

    if 'Variables' not in Problem or not Problem['Variables']:
        Problem['Variables'] = len(Problem['LB'])

    if Problem['Variables'] < 0 or Problem['Variables'] > len(Problem['LB']):
        raise ValueError('Number of variables do not agree with bound constraints')

    if 'Constraints' in Problem and Problem['Constraints']:
        print('MEGA: Constrained tournament handling enabled.')
        Conflag = 1
    else:
        Conflag = 0

    if 'Objectives' not in Problem or not Problem['Objectives']:
        Problem['Objectives'] = 2

    MaxGenerations = get_option('MaxGen', Options, DefaultOpt)
    MaxEvals = get_option('MaxObj', Options, DefaultOpt)
    Pop = get_option('PopSize', Options, DefaultOpt)
    Elite = get_option('Elite', Options, DefaultOpt)

    if Elite:
        eliteinf = max(2, np.ceil(Elite / 2 * Pop))
        elitesup = min(Pop - 2, np.floor((1 - Elite / 2) * Pop))
        print(f'MEGA: MEGA elite size set to the interval {eliteinf} and {elitesup}')

    Tour = get_option('TourSize', Options, DefaultOpt)
    Pc = get_option('Pcross', Options, DefaultOpt)
    Ic = get_option('Icross', Options, DefaultOpt)
    Pm = get_option('Pmut', Options, DefaultOpt) if 'Pmut' in Options else 1 / Problem['Variables']
    Im = get_option('Imut', Options, DefaultOpt)
    Sigma = get_option('Sigma', Options, DefaultOpt) if 'Sigma' in Options else 0
    CTol = get_option('CTol', Options, DefaultOpt)
    CeqTol = get_option('CeqTol', Options, DefaultOpt)
    NormType = get_option('NormType', Options, DefaultOpt)
    NormCons = get_option('NormCons', Options, DefaultOpt)

    Problem['Verbose'] = get_option('Verbosity', Options, DefaultOpt)
    Problem['Tolerance'] = get_option('CPTolerance', Options, DefaultOpt)
    Problem['GenTest'] = get_option('CPGenTest', Options, DefaultOpt)

    import time
    start_time = time.time()

    Problem['Stats'] = {'ObjFunCounter': 0, 'ConCounter': 0, 'GenCounter': 0}
    Problem['Stats']['N1Front'] = []
    Problem['Stats']['NFronts'] = []

    Problem, Population = init_population(Problem, InitialPopulation, Pop, Conflag, CTol, CeqTol, NormType, NormCons,
                                          *varargs)
    Population = rank_population(Population, Elite, Sigma, NormType)

    temp = np.hstack((Population['x'], Population['f'], Population['c'], Population['ceq'],
                      Population['Feasible'].reshape(-1, 1), Population['Rank'].reshape(-1, 1),
                      Population['Fitness'].reshape(-1, 1)))
    temp = temp[np.argsort(temp[:, -1])]

    Population['x'] = temp[:, :Population['x'].shape[1]]
    Population['f'] = temp[:, Population['x'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1]]
    Population['c'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1]:Population['x'].shape[1] +
                                                                                  Population['f'].shape[1] +
                                                                                  Population['c'].shape[1]]
    Population['ceq'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]:
                                Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                Population['ceq'].shape[1]]
    Population['Feasible'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                     Population['ceq'].shape[1]].astype(bool)
    Population['Rank'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                 Population['ceq'].shape[1] + 1].astype(int)
    Population['Fitness'] = temp[:, -1]

    Problem['Stats']['N1Front'].append(np.sum(Population['Rank'] == 1))
    Problem['Stats']['NFronts'].append(np.max(Population['Rank']))

    if Problem['Verbose']:
        print('MEGA is running... ')
        if np.sum(Population['Feasible']) == 0:
            print(
                f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front = {Problem["Stats"]["N1Front"][-1]},  Number of fronts = {Problem["Stats"]["NFronts"][-1]},  All points are unfeasible. Best point: {norm(np.maximum(0, Population["c"][0, :]), NormType) + norm(np.abs(Population["ceq"][0, :]), NormType)}')
        else:
            print(
                f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front = {Problem["Stats"]["N1Front"][-1]},  Number of fronts = {Problem["Stats"]["NFronts"][-1]}')

        if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
            p, pp = draw_illustration(Problem, Population['x'][:, 0], Population['x'][:, 1], Population['f'][:, 0],
                                      Population['f'][:, 1], *varargs)

    while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
        Problem['Stats']['GenCounter'] += 1

        pool = int(np.floor(
            max(eliteinf, min(elitesup, Pop - len(np.where(Population['Rank'][Population['Feasible'] == 1] == 1)[0])))))
        if pool % 2 != 0:
            pool -= 1  # Ensure pool size is even

        parent_chromosome = tournament_selection(Population, int(pool), Tour)

        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)

        Population['x'][Pop - int(pool):Pop, :] = offspring_chromosome['x'][:int(pool), :]

        for i in range(Pop - int(pool), Pop):
            Problem, Population['f'][i, :] = obj_eval(Problem, Population['x'][i, :], *varargs)
            if Conflag:
                Problem, Population['c'][i, :], Population['ceq'][i, :] = con_eval(Problem, Population['x'][i, :],
                                                                                   *varargs)

        if Conflag:
            for i in range(Pop - pool, Pop):
                if NormCons:
                    maxc = np.min(np.maximum(0, Population['c']), axis=0)
                    maxc[maxc == 0] = 1
                    maxceq = np.min(Population['ceq'], axis=0)
                    maxceq[maxceq == 0] = 1
                    Population['Feasible'][i] = (
                            norm(np.maximum(0, Population['c'][i, :]) / maxc, NormType) <= CTol and
                            norm(np.abs(Population['ceq'][i, :]) / np.abs(maxceq), NormType) <= CeqTol
                    )
                else:
                    Population['Feasible'][i] = (
                            norm(np.maximum(0, Population['c'][i, :]), NormType) <= CTol and
                            norm(np.abs(Population['ceq'][i, :]), NormType) <= CeqTol
                    )

                # Debugging: Print feasibility status
                print(f'Individual {i}: c = {Population["c"][i, :]}, ceq = {Population["ceq"][i, :]}, '
                      f'Feasible = {Population["Feasible"][i]}')

        Population = rank_population(Population, Elite, Sigma, NormType)

        temp = np.hstack((Population['x'], Population['f'], Population['c'], Population['ceq'],
                          Population['Feasible'].reshape(-1, 1), Population['Rank'].reshape(-1, 1),
                          Population['Fitness'].reshape(-1, 1)))
        temp = temp[np.argsort(temp[:, -1])]

        Population['x'] = temp[:, :Population['x'].shape[1]]
        Population['f'] = temp[:, Population['x'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1]]
        Population['c'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1]:Population['x'].shape[1] +
                                                                                      Population['f'].shape[1] +
                                                                                      Population['c'].shape[1]]
        Population['ceq'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]:
                                    Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                    Population['ceq'].shape[1]]
        Population['Feasible'] = temp[:,
                                 Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                 Population['ceq'].shape[1]].astype(bool)
        Population['Rank'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] +
                                     Population['ceq'].shape[1] + 1].astype(int)
        Population['Fitness'] = temp[:, -1]

        Problem['Stats']['N1Front'].append(np.sum(Population['Rank'] == 1))
        Problem['Stats']['NFronts'].append(np.max(Population['Rank']))

        if Problem['Verbose']:
            if np.sum(Population['Feasible']) == 0:
                print(
                    f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front = {Problem["Stats"]["N1Front"][-1]},  Number of fronts = {Problem["Stats"]["NFronts"][-1]},  All points are unfeasible. Best point: {norm(np.maximum(0, Population["c"][0, :]), NormType) + norm(np.abs(Population["ceq"][0, :]), NormType)}')
            else:
                print(
                    f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front = {Problem["Stats"]["N1Front"][-1]},  Number of fronts = {Problem["Stats"]["NFronts"][-1]}')
            if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
                update_illustration(p, pp, Problem['Stats']['GenCounter'], Population['x'][:, 0], Population['x'][:, 1],
                                    Population['f'][:, 0], Population['f'][:, 1])

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    if Problem['Stats']['GenCounter'] >= MaxGenerations or Problem['Stats']['ObjFunCounter'] >= MaxEvals:
        print('Maximum number of iterations or objective function evaluations reached')

    NonDomPoint = []
    FrontPoint = {'f': [], 'c': [], 'ceq': []}
    for i in range(Pop):
        if Population['Rank'][i] == 1:
            NonDomPoint.append(Population['x'][i, :])
            FrontPoint['f'].append(Population['f'][i, :])
            FrontPoint['c'].append(Population['c'][i, :])
            FrontPoint['ceq'].append(Population['ceq'][i, :])

    if Problem['Verbose']:
        if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
            terminate_illustration(p, pp, np.array(NonDomPoint)[:, 0], np.array(NonDomPoint)[:, 1],
                                   np.array(FrontPoint['f'])[:, 0], np.array(FrontPoint['f'])[:, 1], len(NonDomPoint))
        print(Problem['Stats'])

    return np.array(NonDomPoint), {key: np.array(value) for key, value in FrontPoint.items()}, Problem['Stats']


def init_population(Problem, InitialPopulation, Size, Conflag, CTol, CeqTol, NormType, NormCons, *varargs):
    Population = {
        'x': np.zeros((Size, Problem['Variables'])),
        'f': np.zeros((Size, Problem['Objectives'])),
        'c': np.zeros((Size, (Problem.get('c_dim', [])))),
        'ceq': np.zeros((Size, (Problem.get('ceq_dim', [])))),
        'Feasible': np.zeros(Size, dtype=bool),
        'Rank': np.zeros(Size),
        'Fitness': np.zeros(Size)
    }

    if InitialPopulation:
        for i in range(len(InitialPopulation)):
            Population['x'][i, :] = np.clip(InitialPopulation[i], Problem['LB'], Problem['UB'])
            Problem, Population['f'][i, :] = obj_eval(Problem, Population['x'][i, :], *varargs)
            if Conflag:
                Problem, Population['c'][i, :], Population['ceq'][i, :] = con_eval(Problem, Population['x'][i, :],
                                                                                   *varargs)
                Population['Feasible'][i] = (
                        norm(np.maximum(0, Population['c'][i, :]), NormType) <= CTol and
                        norm(np.abs(Population['ceq'][i, :]), NormType) <= CeqTol
                )
                print(f"Initial individual {i} feasibility: {Population['Feasible'][i]}")

    for i in range(len(InitialPopulation), Size):
        Population['x'][i, :] = Problem['LB'] + (Problem['UB'] - Problem['LB']) * np.random.rand(Problem['Variables'])
        Problem, Population['f'][i, :] = obj_eval(Problem, Population['x'][i, :], *varargs)
        if Conflag:
            Problem, Population['c'][i, :], Population['ceq'][i, :] = con_eval(Problem, Population['x'][i, :], *varargs)
            Population['Feasible'][i] = (
                    norm(np.maximum(0, Population['c'][i, :]), NormType) <= CTol and
                    norm(np.abs(Population['ceq'][i, :]), NormType) <= CeqTol
            )
            print(f"Generated individual {i} feasibility: {Population['Feasible'][i]}")

    return Problem, Population


def rank_population(Population, elite, sigma, NormType):
    pop_size = Population['f'].shape[0]
    num_obj = Population['f'].shape[1]

    IP = np.where(Population['Feasible'] == 1)[0]
    if len(IP) == 0:
        print('Feasibility check: No feasible solutions found in the population.')
        raise ValueError('No feasible solutions found in the population.')

    P = np.hstack((Population['f'][IP], IP.reshape(-1, 1)))
    rank = 1
    while P.shape[0] > 0:
        ND = nondom(P, num_obj)
        P = np.array([row for row in P if row.tolist() not in ND.tolist()])
        for i in range(ND.shape[0]):
            Population['Rank'][int(ND[i, -1])] = rank
        rank += 1

    I = np.where(Population['Rank'] == 1)[0]
    if len(I) == 0:
        raise ValueError('No individuals in the first front.')

    ideal = np.min(Population['f'][I, :], axis=0)
    if sigma == 0:
        nadir = np.max(Population['f'][I, :], axis=0)
        dnorm = norm(nadir - ideal)
        if dnorm == 0:
            dnorm = norm(np.max(Population['f'], axis=0) - np.min(Population['f'], axis=0))
        sigma = 2 * dnorm * (pop_size - np.floor(elite * pop_size / 2) - 1) ** (-1 / (num_obj - 1))

    fk = 1
    if sigma != 0:
        frente = 1
        while frente < rank:
            I = np.where(Population['Rank'] == frente)[0]
            LI = len(I)
            for i in range(LI):
                if not np.any(I[i] == np.where(Population['f'][I, :] == ideal)[1]):
                    nc = 0
                    for j in range(LI):
                        nc += share(norm(Population['f'][I[i], :] - Population['f'][I[j], :]), sigma)
                    Population['Fitness'][I[i]] = fk * nc
                else:
                    Population['Fitness'][I[i]] = fk
            fk = np.floor(np.max(Population['Fitness'][I]) + 1)
            frente += 1
    else:
        Population['Fitness'] = Population['Rank']

    IP = np.where(Population['Feasible'] == 0)[0]
    for i in IP:
        Population['Rank'][i] = rank
        Population['Fitness'][i] = fk + norm(np.maximum(0, Population['c'][i, :]), NormType) + norm(
            np.abs(Population['ceq'][i, :]), NormType)

    return Population


def tournament_selection(chromosomes, pool_size, tour_size):
    pop_size = chromosomes['x'].shape[0]
    P = {'x': np.zeros((pool_size, chromosomes['x'].shape[1]))}
    for i in range(pool_size):
        candidate = np.random.choice(pop_size, tour_size, replace=False)
        fitness = chromosomes['Fitness'][candidate]
        min_candidate = candidate[np.argmin(fitness)]
        P['x'][i, :] = chromosomes['x'][min_candidate, :]
    return P


def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome['x'].shape

    if N % 2 != 0:
        N -= 1

    child = np.zeros((N, V))
    p = 0
    while p < N:
        parent_1 = parent_chromosome['x'][np.random.randint(N), :]
        parent_2 = parent_chromosome['x'][np.random.randint(N), :]
        while np.array_equal(parent_1, parent_2):
            parent_2 = parent_chromosome['x'][np.random.randint(N), :]
        if np.random.rand() < pc:
            child_1 = np.zeros(V)
            child_2 = np.zeros(V)
            for j in range(V):
                u = np.random.rand()
                if u <= 0.5:
                    bq = (2 * u) ** (1 / (mu + 1))
                else:
                    bq = (1 / (2 * (1 - u))) ** (1 / (mu + 1))
                child_1[j] = 0.5 * ((1 + bq) * parent_1[j] + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * ((1 - bq) * parent_1[j] + (1 + bq) * parent_2[j])
            child_1 = bounds(child_1, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            child_2 = bounds(child_2, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
        else:
            child_1 = parent_1
            child_2 = parent_2
        if np.random.rand() < np.sqrt(pm):
            for j in range(V):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_1[j] = child_1[j] + (Problem['UB'][j] - Problem['LB'][j]) * delta
        child_1 = bounds(child_1, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
        if np.random.rand() < np.sqrt(pm):
            for j in range(V):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_2[j] = child_2[j] + (Problem['UB'][j] - Problem['LB'][j]) * delta
        child_2 = bounds(child_2, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
        child[p, :] = child_1
        child[p + 1, :] = child_2
        p += 2
    P = {'x': child}
    return P


def bounds(X, L, U):
    return np.maximum(np.minimum(X, U), L)


def obj_eval(Problem, x, *varargs):
    try:
        ObjValue = Problem['ObjFunction'](x)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise ValueError(
            f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return Problem, ObjValue


def con_eval(Problem, x, *varargs):
    Problem['Stats']['ConCounter'] += 1
    try:
        c, ceq = Problem['Constraints'](x)
        c = c if c is not None else np.array([])
        ceq = ceq if ceq is not None else np.array([])

        # Debugging: Print constraints values
        print(f'Constraints evaluation: c = {c}, ceq = {ceq}')

    except Exception as e:
        raise ValueError(
            f'Cannot continue because user supplied function constraints failed with the following error:\n{e}')
    return Problem, c, ceq



def nondom(P, nv):
    n = P.shape[0]
    m = P.shape[1] - nv
    k = 0
    i = 0
    PL = []
    while i < n:
        cand = 1
        j = 0
        while j < n and cand == 1:
            if j != i and dom(P[j, :m], P[i, :m]):
                cand = 0
            j += 1
        if cand == 1:
            PL.append(P[i, :])
            k += 1
        i += 1
    return np.array(PL)


def share(dist, sigma):
    if dist <= sigma:
        return 1 - (dist / sigma) ** 2
    else:
        return 0


def dom(x, y):
    if np.all(x <= y) and np.any(x < y):
        return True
    return False


def draw_illustration(Problem, X, Y, F1, F2, *varargs):
    plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax1.set_title('Objective function')
    xx, yy = np.meshgrid(np.linspace(Problem['LB'][0], Problem['UB'][0], 80),
                         np.linspace(Problem['LB'][1], Problem['UB'][1], 80))
    z = np.zeros((xx.shape[0], xx.shape[1], 2))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            f = Problem['ObjFunction']([xx[i, j], yy[i, j]], *varargs)
            z[i, j, 0] = f[0]
            z[i, j, 1] = f[1]
    ax1.plot_surface(xx, yy, z[:, :, 0], alpha=0.5, cmap='viridis')
    ax1.plot_surface(xx, yy, z[:, :, 1], alpha=0.5, cmap='plasma')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f1(x) and f2(x)')

    ax2 = plt.subplot(2, 2, 2)
    ax2.contour(xx, yy, z[:, :, 0], cmap='viridis')
    ax2.contour(xx, yy, z[:, :, 1], cmap='plasma')
    ax2.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    p, = ax2.plot(X, Y, 'b.', label='Population')

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(z[:, :, 0].flatten(), z[:, :, 1].flatten(), 'b.')
    ax3.set_xlabel('f1(x)')
    ax3.set_ylabel('f2(x)')

    ax4 = plt.subplot(2, 2, 4)
    ax4.grid(True)
    ax4.set_xlabel('f1(x)')
    ax4.set_ylabel('f2(x)')
    ax4.set_title(f'Population at generation: {0}')
    pp, = ax4.plot(F1, F2, 'b.', label='Pareto Front')

    plt.draw()
    plt.tight_layout()
    return p, pp


def update_illustration(p, pp, gen, X, Y, F1, F2):
    p.set_xdata(X)
    p.set_ydata(Y)
    pp.set_xdata(F1)
    pp.set_ydata(F2)
    plt.title(f'Population at generation: {gen}')
    plt.draw()


def terminate_illustration(p, pp, X, Y, F1, F2, ND):
    plt.title(f'Search terminated. Number of nondominated solutions: {ND}')
    p.set_xdata(X)
    p.set_ydata(Y)
    pp.set_xdata(F1)
    pp.set_ydata(F2)
    plt.draw()
