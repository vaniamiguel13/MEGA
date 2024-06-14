import numpy as np
from copy import deepcopy
from collections import namedtuple

# Define a namedtuple for individuals in the population
Individual = namedtuple('Individual', ['x', 'f', 'c', 'ceq', 'Feasible', 'Rank', 'Fitness'])

import numpy as np
from copy import deepcopy
from collections import namedtuple

# Define a namedtuple for individuals in the population
Individual = namedtuple('Individual', ['x', 'f', 'c', 'ceq', 'Feasible', 'Rank', 'Fitness'])

def mega_con(problem, initial_population, options, *args):
    """
    MEGAcon: Multiobjective Elitist Genetic Algorithm for nonlinear multiobjective function minimization
    with bound constraints and nonlinear constraints.
    """
    # Default options
    default_opt = {
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
        'CPTolerance': 1e-6,
        'CPGenTest': 0.01,
        'CTol': 1e-2,
        'CeqTol': 1e-2,
        'NormType': np.inf,
        'NormCons': 1,
        'Verbosity': 0
    }

    # Check parameters
    if not isinstance(problem, dict):
        raise ValueError("First parameter must be a dict.")

    if 'ObjFunction' not in problem or not problem['ObjFunction']:
        raise ValueError("Objective function name is missing.")

    if 'LB' not in problem or 'UB' not in problem or not np.isfinite(problem['LB']).all() or not np.isfinite(problem['UB']).all():
        raise ValueError("Population relies on finite bounds on all variables.")

    problem['LB'] = np.array(problem['LB'])
    problem['UB'] = np.array(problem['UB'])

    if len(problem['LB']) != len(problem['UB']):
        raise ValueError("Lower bound and upper bound arrays length mismatch.")

    if 'Variables' not in problem or not problem['Variables']:
        problem['Variables'] = len(problem['LB'])
    elif problem['Variables'] < 0 or problem['Variables'] > len(problem['LB']):
        raise ValueError("Number of variables do not agree with bound constraints")

    if 'Constraints' in problem and problem['Constraints']:
        conflag = 1
        print("MEGA: Constrained tournament handling enabled.")
    else:
        conflag = 0

    if 'Objectives' not in problem or not problem['Objectives']:
        problem['Objectives'] = 2
    elif problem['Objectives'] < 0 or problem['Objectives'] > len(problem['LB']):
        raise ValueError("Number of variables do not agree with bound constraints")

    # Initialize options
    options = {**default_opt, **(options or {})}

    max_generations = options['MaxGen']
    max_evals = options['MaxObj']
    pop_size = options['PopSize']
    elite_prop = options['Elite']
    tour_size = options['TourSize']
    pc = options['Pcross']
    ic = options['Icross']
    if 'Pmut' not in options:
        pm = 1 / problem['Variables']
        print(f"MEGA: MEGA mutation probability set to {pm}")
    else:
        pm = options['Pmut']
    im = options['Imut']
    if 'Sigma' not in options:
        print("MEGA: MEGA niching radius will be adapted during the search")
        sigma = 0
    else:
        sigma = options['Sigma']
    ctol = options['CTol']
    ceqtol = options['CeqTol']
    norm_type = options['NormType']
    norm_cons = options['NormCons']

    # Global options
    problem['Verbose'] = options['Verbosity']
    problem['Tolerance'] = options['CPTolerance']
    problem['GenTest'] = options['CPGenTest']

    # Initialize counters
    problem['Stats'] = {
        'ObjFunCounter': 0,
        'ConCounter': 0,
        'GenCounter': 0
    }

    # Initialize population
    population = init_population(problem, initial_population, pop_size, conflag, ctol, ceqtol, norm_type, norm_cons, *args)
    population = rank_population(population, elite_prop, sigma, norm_type)

    # Sort population
    population = sort_population(population)

    # Initialize statistics
    problem['Stats']['N1Front'] = [len([p for p in population if p.Rank == 1])]
    problem['Stats']['NFronts'] = [max([p.Rank for p in population])]

    if problem['Verbose']:
        feasible_points = [p for p in population if p.Feasible]
        if not feasible_points:
            best_point = "N/A"
            if len(population) > 0:
                best_point = norm(np.maximum(0, np.array(population[0].c)), norm_type) + norm(np.abs(np.array(population[0].ceq)), norm_type)
            print(
                f"Gen: {problem['Stats']['GenCounter'] + 1}  No. points in 1st front = {problem['Stats']['N1Front'][-1]}  Number of fronts = {problem['Stats']['NFronts'][-1]}  All points are unfeasible. Best point: {best_point}")
        else:
            print(
                f"Gen: {problem['Stats']['GenCounter'] + 1}  No. points in 1st front = {problem['Stats']['N1Front'][-1]}  Number of fronts = {problem['Stats']['NFronts'][-1]}")

    # Main cycle of the genetic algorithm
    while problem['Stats']['GenCounter'] < max_generations and problem['Stats']['ObjFunCounter'] < max_evals:
        # Increment generation counter
        problem['Stats']['GenCounter'] += 1

        # Select parents
        if elite_prop:
            feasible_rank_1 = len([p for p in population if p.Feasible and p.Rank == 1])
            elite_inf = max(2, int(np.ceil(elite_prop / 2 * pop_size)))
            elite_sup = min(pop_size - 2, int(np.floor((1 - elite_prop / 2) * pop_size)))
            pool_size = int(max(elite_inf, min(elite_sup, pop_size - feasible_rank_1)))
        else:
            pool_size = pop_size

        parent_chromosomes = tournament_selection(population, pool_size, tour_size)

        # Perform crossover and mutation
        offspring_chromosomes = genetic_operator(problem, parent_chromosomes, pc, pm, ic, im)

        # Evaluate offspring
        for i in range(pool_size):
            problem, offspring_chromosomes['f'][i] = obj_eval(problem, offspring_chromosomes['x'][i], *args)
            if conflag:
                problem, offspring_chromosomes['c'][i], offspring_chromosomes['ceq'][i] = con_eval(problem, offspring_chromosomes['x'][i], *args)
                offspring_chromosomes['Feasible'][i] = is_feasible(offspring_chromosomes['c'][i], offspring_chromosomes['ceq'][i], norm_type, ctol, ceqtol, norm_cons)
            else:
                offspring_chromosomes['c'][i] = np.zeros_like(offspring_chromosomes['x'][i])
                offspring_chromosomes['ceq'][i] = np.zeros_like(offspring_chromosomes['x'][i])
                offspring_chromosomes['Feasible'][i] = True

        # Replace worst chromosomes with offspring
        population[-pool_size:] = [Individual(offspring_chromosomes['x'][x], offspring_chromosomes['f'][x], offspring_chromosomes['c'][x], offspring_chromosomes['ceq'][x], offspring_chromosomes['Feasible'][x], 0, 0) for x in range(pool_size)]

        # Rank population
        population = rank_population(population, elite_prop, sigma, norm_type)

        # Sort population
        population = sort_population(population)

        # Update statistics
        problem['Stats']['N1Front'].append(len([p for p in population if p.Rank == 1]))
        problem['Stats']['NFronts'].append(max([p.Rank for p in population]))

        if problem['Verbose']:
            feasible_points = [p for p in population if p.Feasible]
            if not feasible_points:
                best_point = "N/A"
                if len(population) > 0:
                    best_point = norm(np.maximum(0, np.array(population[0].c)), norm_type) + norm(np.abs(np.array(population[0].ceq)), norm_type)
                print(
                    f"Gen: {problem['Stats']['GenCounter'] + 1}  No. points in 1st front = {problem['Stats']['N1Front'][-1]}  Number of fronts = {problem['Stats']['NFronts'][-1]}  All points are unfeasible. Best point: {best_point}")
            else:
                print(
                    f"Gen: {problem['Stats']['GenCounter'] + 1}  No. points in 1st front = {problem['Stats']['N1Front'][-1]}  Number of fronts = {problem['Stats']['NFronts'][-1]}")

    # Return non-dominated points and run data
    non_dom_points = [p.x for p in population if p.Rank == 1]
    front_points = {
        'f': [p.f for p in population if p.Rank == 1],
        'c': [p.c for p in population if p.Rank == 1],
        'ceq': [p.ceq for p in population if p.Rank == 1]
    }
    run_data = problem['Stats']

    if problem['Verbose']:
        print(problem['Stats'])

    return non_dom_points, front_points, run_data



def init_population(problem, initial_population, size, conflag, ctol, ceqtol, norm_type, norm_cons, *args):
    """
    Initialize population
    """
    population = []

    if initial_population:
        if not isinstance(initial_population, list):
            raise ValueError("Initial population must be a list of dicts.")

        if len(initial_population) > size:
            raise ValueError("Initial population size must be less than PopSize.")

        for individual in initial_population:
            x = np.clip(individual['x'], problem['LB'], problem['UB'])
            problem, f = obj_eval(problem, x, *args)
            if conflag:
                problem, c, ceq = con_eval(problem, x, *args)
                feasible = is_feasible(c, ceq, norm_type, ctol, ceqtol, norm_cons)
            else:
                c = np.zeros_like(x)
                ceq = np.zeros_like(x)
                feasible = True
            population.append(Individual(x, f, c, ceq, feasible, 0, 0))

    for _ in range(len(initial_population), size):
        x = problem['LB'] + (problem['UB'] - problem['LB']) * np.random.rand(problem['Variables'])
        problem, f = obj_eval(problem, x, *args)
        if conflag:
            problem, c, ceq = con_eval(problem, x, *args)
            feasible = is_feasible(c, ceq, norm_type, ctol, ceqtol, norm_cons)
        else:
            c = np.zeros_like(x)
            ceq = np.zeros_like(x)
            feasible = True
        population.append(Individual(x, f, c, ceq, feasible, 0, 0))

    return population


def share(dist, sigma):
    """
    Sharing function
    """
    if dist <= sigma:
        return 1 - (dist / sigma) ** 2
    else:
        return 0
def rank_population(population, elite_prop, sigma, norm_type):
    """
    Compute non-dominated sorting and niching
    """
    pop_size = len(population)
    num_obj = len(population[0].f)

    # Compute rank
    feasible_indices = [i for i, p in enumerate(population) if p.Feasible]
    p = [[p.f, i] for i, p in enumerate(population) if p.Feasible]
    rank = 1
    while p:
        non_dom = non_dominated(p, num_obj)
        p = [point for point in p if point not in non_dom]
        for point in non_dom:
            population[point[1]] = population[point[1]]._replace(Rank=rank)
        rank += 1

    feasible_rank_1 = [i for i in feasible_indices if population[i].Rank == 1]
    if feasible_rank_1:
        ideal = np.min([population[i].f for i in feasible_rank_1], axis=0)
        if sigma == 0:
            nadir = np.max([population[i].f for i in feasible_rank_1], axis=0)
            dnorm = norm(nadir - ideal, norm_type)
            if dnorm == 0:
                dnorm = norm(np.max([p.f for p in population], axis=0) - np.min([p.f for p in population], axis=0), norm_type)
            sigma = 2 * dnorm * (pop_size - int(np.floor(elite_prop * pop_size / 2))) ** (-1 / (num_obj - 1))

    # Compute sharing values
    fk = 1
    if sigma != 0:
        front = 1
        while front < rank:
            indices = [i for i, p in enumerate(population) if p.Rank == front]
            for i in indices:
                if i not in feasible_rank_1:
                    nc = sum(share(norm(population[i].f - population[j].f, norm_type), sigma) for j in indices)
                    population[i] = population[i]._replace(Fitness=fk * nc)
                else:
                    population[i] = population[i]._replace(Fitness=fk)
            fk = int(max([population[i].Fitness for i in indices]) + 1)
            front += 1
    else:
        for i in range(len(population)):
            population[i] = population[i]._replace(Fitness=population[i].Rank)

    # Unfeasible points
    for i, p in enumerate(population):
        if not p.Feasible:
            population[i] = population[i]._replace(Rank=rank, Fitness=fk + norm(np.maximum(0, p.c), norm_type) + norm(np.abs(p.ceq), norm_type))

    return population




def tournament_selection(population, pool_size, tour_size):
    """
    Tournament selection
    """
    pop_size = len(population)
    parent_chromosomes = {'x': [], 'f': [], 'c': [], 'ceq': [], 'Feasible': [], 'Rank': [], 'Fitness': []}

    for _ in range(pool_size):
        candidates = np.random.randint(pop_size, size=tour_size)
        fitnesses = [population[i].Fitness for i in candidates]
        best_candidate = candidates[np.argmin(fitnesses)]
        parent_chromosomes['x'].append(population[best_candidate].x)
        parent_chromosomes['f'].append(population[best_candidate].f)
        parent_chromosomes['c'].append(population[best_candidate].c)
        parent_chromosomes['ceq'].append(population[best_candidate].ceq)
        parent_chromosomes['Feasible'].append(population[best_candidate].Feasible)
        parent_chromosomes['Rank'].append(population[best_candidate].Rank)
        parent_chromosomes['Fitness'].append(population[best_candidate].Fitness)

    return parent_chromosomes


def genetic_operator(problem, parent_chromosomes, pc, pm, mu, mum):
    """
    Perform crossover and mutation
    """
    n, v = len(parent_chromosomes['x']), problem['Variables']
    child = {'x': np.zeros((n, v)), 'f': np.zeros((n, len(parent_chromosomes['f'][0]))),
             'c': np.zeros((n, len(parent_chromosomes['c'][0]))), 'ceq': np.zeros((n, len(parent_chromosomes['ceq'][0]))),
             'Feasible': np.zeros(n, dtype=bool), 'Rank': np.zeros(n, dtype=int), 'Fitness': np.zeros(n)}

    for p in range(0, n, 2):
        parent_1_idx = np.random.randint(n)
        parent_2_idx = np.random.randint(n)
        while parent_2_idx == parent_1_idx:
            parent_2_idx = np.random.randint(n)

        parent_1 = parent_chromosomes['x'][parent_1_idx]
        parent_2 = parent_chromosomes['x'][parent_2_idx]

        # SBX Crossover
        if np.random.rand() < pc:
            child_1 = np.zeros(v)
            child_2 = np.zeros(v)
            for j in range(v):
                u = np.random.rand()
                if u <= 0.5:
                    bq = (2 * u) ** (1 / (mu + 1))
                else:
                    bq = (1 / (2 * (1 - u))) ** (1 / (mu + 1))
                child_1[j] = 0.5 * ((1 + bq) * parent_1[j] + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * ((1 - bq) * parent_1[j] + (1 + bq) * parent_2[j])
            child_1 = np.clip(child_1, problem['LB'], problem['UB'])
            child_2 = np.clip(child_2, problem['LB'], problem['UB'])
        else:
            child_1 = parent_1.copy()
            child_2 = parent_2.copy()

        # Polynomial Mutation
        if np.random.rand() < np.sqrt(pm):
            for j in range(v):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_1[j] += (problem['UB'][j] - problem['LB'][j]) * delta
            child_1 = np.clip(child_1, problem['LB'], problem['UB'])

        if np.random.rand() < np.sqrt(pm):
            for j in range(v):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_2[j] += (problem['UB'][j] - problem['LB'][j]) * delta
            child_2 = np.clip(child_2, problem['LB'], problem['UB'])

        child['x'][p] = child_1
        child['x'][p + 1] = child_2

    return child
def is_feasible(c, ceq, norm_type, ctol, ceqtol, norm_cons):
    """
    Check if a point is feasible
    """
    # Ensure c and ceq are arrays
    c = np.array(c)
    ceq = np.array(ceq)

    if norm_cons:
        max_c = np.min(np.maximum(0, c), axis=0)
        if np.isscalar(max_c):
            max_c = np.array([max_c])
        if isinstance(max_c, np.ndarray):  # Ensure max_c is an array
            max_c[max_c == 0] = 1
        max_ceq = np.min(np.abs(ceq), axis=0)
        if np.isscalar(max_ceq):
            max_ceq = np.array([max_ceq])
        if isinstance(max_ceq, np.ndarray):  # Ensure max_ceq is an array
            max_ceq[max_ceq == 0] = 1
        return norm(np.maximum(0, c) / max_c, norm_type) <= ctol and norm(np.abs(ceq) / max_ceq, norm_type) <= ceqtol
    else:
        return norm(np.maximum(0, c), norm_type) <= ctol and norm(np.abs(ceq), norm_type) <= ceqtol


def obj_eval(problem, x, *args):
    """
    Evaluate objective function
    """
    try:
        obj_value = problem['ObjFunction'](x, *args)
        problem['Stats']['ObjFunCounter'] += 1
        return problem, obj_value
    except Exception as e:
        raise ValueError(f"Cannot continue because user supplied objective function failed with the following error:\n{e}")


def con_eval(problem, x, *args):
    """
    Evaluate constraint functions
    """
    problem['Stats']['ConCounter'] += 1
    try:
        c, ceq = problem['Constraints'](x, *args)
        if c is None:
            c = np.zeros_like(x)
        if ceq is None:
            ceq = np.zeros_like(x)
        return problem, c, ceq
    except Exception as e:
        raise ValueError(f"Cannot continue because user supplied function constraints failed with the following error:\n{e}")


def non_dominated(p, num_vars):
    """
    Find non-dominated points
    """
    n = len(p)
    non_dom = []
    i = 0
    while i < n:
        cand = True
        j = 0
        while j < n and cand:
            if j != i and dominates(p[j][0], p[i][0]):
                cand = False
            j += 1
        if cand:
            non_dom.append(p[i])
        i += 1
    return non_dom


def dominates(x, y):
    """
    Check if x dominates y
    """
    m = len(x)
    i = 0
    while i < m:
        if x[i] > y[i]:
            return False
        i += 1
    i = 0
    while i < m:
        if x[i] < y[i]:
            return True
        i += 1
    return False


def sort_population(population):
    """
    Sort population based on fitness
    """
    sorted_population = sorted(
        population,
        key=lambda p: p.Fitness, reverse=True
    )
    return sorted_population


def norm(x, p=np.inf):
    """
    Compute the p-norm of a vector
    """
    return np.sum(np.abs(x) ** p) ** (1 / p)
