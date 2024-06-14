import numpy as np
from copy import deepcopy
from collections import namedtuple

Individual = namedtuple('Individual',
                        ['x', 'f', 'c', 'ceq', 'Feasible', 'Rank', 'Fitness', 'DominatedSet', 'DominationCount',
                         'CrowdingDistance'])


class Individual:
    def __init__(self, x, f, c, ceq, feasible, rank=0, fitness=0.0):
        self.x = x
        self.f = f
        self.c = c
        self.ceq = ceq
        self.Feasible = feasible
        self.Rank = rank
        self.Fitness = fitness
        self.DominatedSet = []
        self.DominationCount = 0
        self.CrowdingDistance = 0.0

def create_individual(x, f, c, ceq, feasible):
    return Individual(x, f, c, ceq, feasible)


import numpy as np
from copy import deepcopy
from collections import namedtuple

import pandas as pd

def mega_con(problem, initial_population, options, *args):
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

    problem['Verbose'] = options['Verbosity']
    problem['Tolerance'] = options['CPTolerance']
    problem['GenTest'] = options['CPGenTest']

    problem['Stats'] = {
        'ObjFunCounter': 0,
        'ConCounter': 0,
        'GenCounter': 0,
        'N1Front': [],
        'NFronts': []
    }

    population = init_population(problem, initial_population, pop_size, conflag, ctol, ceqtol, norm_type, norm_cons, *args)
    population, num_fronts, n1fronts = rank_population(population)

    population = sort_population(population)

    problem['Stats']['N1Front'].append(n1fronts[0])
    problem['Stats']['NFronts'].append(num_fronts)

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

    while problem['Stats']['GenCounter'] < max_generations and problem['Stats']['ObjFunCounter'] < max_evals:
        problem['Stats']['GenCounter'] += 1

        if elite_prop:
            feasible_rank_1 = len([p for p in population if p.Feasible and p.Rank == 1])
            elite_inf = max(2, int(np.ceil(elite_prop / 2 * pop_size)))
            elite_sup = min(pop_size - 2, int(np.floor((1 - elite_prop / 2) * pop_size)))
            pool_size = int(max(elite_inf, min(elite_sup, pop_size - feasible_rank_1)))
        else:
            pool_size = pop_size

        parent_chromosomes = tournament_selection(population, pool_size, tour_size)

        offspring_chromosomes = genetic_operator(problem, parent_chromosomes, pc, pm, ic, im)

        for i in range(pool_size):
            problem, offspring_chromosomes['f'][i] = obj_eval(problem, offspring_chromosomes['x'][i], *args)
            if conflag:
                problem, offspring_chromosomes['c'][i], offspring_chromosomes['ceq'][i] = con_eval(problem, offspring_chromosomes['x'][i], *args)
                offspring_chromosomes['Feasible'][i] = is_feasible(offspring_chromosomes['c'][i], offspring_chromosomes['ceq'][i], norm_type, ctol, ceqtol, norm_cons)
            else:
                offspring_chromosomes['c'][i] = np.zeros_like(offspring_chromosomes['x'][i])
                offspring_chromosomes['ceq'][i] = np.zeros_like(offspring_chromosomes['x'][i])
                offspring_chromosomes['Feasible'][i] = True

        population[-pool_size:] = [create_individual(offspring_chromosomes['x'][x], offspring_chromosomes['f'][x], offspring_chromosomes['c'][x], offspring_chromosomes['ceq'][x], offspring_chromosomes['Feasible'][x]) for x in range(pool_size)]

        population, num_fronts, n1fronts = rank_population(population)

        population = sort_population(population)

        problem['Stats']['N1Front'].append(n1fronts[0])
        problem['Stats']['NFronts'].append(num_fronts)

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

    non_dom_points = [p.x for p in population if p.Rank == 1]
    front_points = {
        'f': [p.f for p in population if p.Rank == 1],
        'c': [p.c for p in population if p.Rank == 1],
        'ceq': [p.ceq for p in population if p.Rank == 1]
    }
    run_data = problem['Stats']

    if problem['Verbose']:
        print(problem['Stats'])

    results_df = pd.DataFrame({
        'NonDominatedPoints': non_dom_points,
        'ObjectiveValues': front_points['f'],
        'ConstraintValues': front_points['c'],
        'EqualityConstraintValues': front_points['ceq']
    })

    return results_df, run_data



def init_population(problem, initial_population, size, conflag, ctol, ceqtol, norm_type, norm_cons, *args):
    population = []
    if initial_population:
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
            population.append(create_individual(x, f, c, ceq, feasible))

    while len(population) < size:
        x = problem['LB'] + (problem['UB'] - problem['LB']) * np.random.rand(problem['Variables'])
        problem, f = obj_eval(problem, x, *args)
        if conflag:
            problem, c, ceq = con_eval(problem, x, *args)
            feasible = is_feasible(c, ceq, norm_type, ctol, ceqtol, norm_cons)
        else:
            c = np.zeros_like(x)
            ceq = np.zeros_like(x)
            feasible = True
        population.append(create_individual(x, f, c, ceq, feasible))

    return population


def share(dist, sigma):
    """
    Sharing function
    """
    if dist <= sigma:
        return 1 - (dist / sigma) ** 2
    else:
        return 0
def rank_population(population):
    fronts = [[]]
    for p in population:
        p.DominatedSet = []
        p.DominationCount = 0
        p.Rank = 0
        for q in population:
            if dominates(p.f, q.f):
                p.DominatedSet.append(q)
            elif dominates(q.f, p.f):
                p.DominationCount += 1
        if p.DominationCount == 0:
            p.Rank = 1
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.DominatedSet:
                q.DominationCount -= 1
                if q.DominationCount == 0:
                    q.Rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # Remove the empty last front

    # Handle the case where all individuals are non-dominated
    if len(fronts) == 1:
        for front in fronts:
            calculate_crowding_distance(front)
            for p in front:
                p.Fitness = p.CrowdingDistance
        return population, 1, [len(fronts[0])]

    # Proceed with the regular ranking process
    for front in fronts:
        calculate_crowding_distance(front)
        for p in front:
            p.Fitness = p.CrowdingDistance

    return population, len(fronts), [len(front) for front in fronts]

def calculate_crowding_distance(front):
    num_solutions = len(front)
    if num_solutions == 0:
        return
    num_objectives = len(front[0].f)

    for p in front:
        p.CrowdingDistance = 0

    for m in range(num_objectives):
        front.sort(key=lambda x: x.f[m])
        front[0].CrowdingDistance = float('inf')
        front[-1].CrowdingDistance = float('inf')
        m_values = [p.f[m] for p in front]
        scale = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        for i in range(1, num_solutions - 1):
            front[i].CrowdingDistance += (front[i + 1].f[m] - front[i - 1].f[m]) / scale

def dominates(p, q):
    and_condition = np.all(np.array(p) <= np.array(q))
    or_condition = np.any(np.array(p) < np.array(q))
    return and_condition and or_condition

def sort_population(population):
    sorted_population = sorted(
        population,
        key=lambda x: (x.Rank, -x.CrowdingDistance)
    )
    return sorted_population




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
             'c': np.zeros((n, len(parent_chromosomes['c'][0]))),
             'ceq': np.zeros((n, len(parent_chromosomes['ceq'][0]))),
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
    c = np.array(c)
    ceq = np.array(ceq)

    if norm_cons:
        max_c = np.maximum(0, c)
        max_ceq = np.abs(ceq)
        max_c[max_c == 0] = 1e-10  # Avoid division by zero
        max_ceq[max_ceq == 0] = 1e-10  # Avoid division by zero
        return norm(max_c / max_c, norm_type) <= ctol and norm(max_ceq / max_ceq, norm_type) <= ceqtol
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
        raise ValueError(
            f"Cannot continue because user supplied objective function failed with the following error:\n{e}")


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
        raise ValueError(
            f"Cannot continue because user supplied function constraints failed with the following error:\n{e}")


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


def norm(x, p=np.inf):
    """
    Compute the p-norm of a vector
    """
    return np.sum(np.abs(x) ** p) ** (1 / p)
