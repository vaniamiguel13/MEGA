import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import math


def InitPopulation(Problem, InitialPopulation, Size, *args):
    Population = {'x': [], 'f': []}
    if InitialPopulation:
        if len(InitialPopulation) > Size:
            raise ValueError('Initial population size must be inferior to PopSize.')
        for ind in InitialPopulation:
            x = np.clip(ind['x'], Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            Population['x'].append(x)
            _, f = ObjEval(Problem, x, *args)
            Population['f'].append(f)
    for i in range(len(InitialPopulation), Size):
        x = Problem['LB'][:Problem['Variables']] + (
                    Problem['UB'][:Problem['Variables']] - Problem['LB'][:Problem['Variables']]) * np.random.rand(
            Problem['Variables'])
        Population['x'].append(x)
        _, f = ObjEval(Problem, x, *args)
        Population['f'].append(f)
    Population['x'] = np.array(Population['x'])
    Population['f'] = np.array(Population['f'])
    return Problem, Population

def ObjEval(Problem, x, *args):
    try:
        ObjValue = Problem['ObjFunction'](x, *args)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise RuntimeError(f"User supplied objective function failed with the following error:\n{str(e)}")
    return Problem, ObjValue

def Bounds(X, L, U):
    return np.clip(X, L, U)

def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes['x'].shape[0]
    P = {'x': [], 'f': []}
    for _ in range(pool_size):
        candidates = np.random.choice(pop, tour_size, replace=False)
        best = candidates[np.argmin(chromosomes['f'][candidates])]
        P['x'].append(chromosomes['x'][best])
        P['f'].append(chromosomes['f'][best])
    P['x'] = np.array(P['x'])
    P['f'] = np.array(P['f'])
    return P

def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome['x'].shape
    child = np.zeros((N, V))
    p = 0
    while p < N:
        parent_1 = parent_chromosome['x'][np.random.randint(N)]
        parent_2 = parent_chromosome['x'][np.random.randint(N)]
        if np.random.rand() < pc:
            for j in range(V):
                u = np.random.rand()
                bq = (2 * u) ** (1 / (mu + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (mu + 1))
                child_1 = 0.5 * (((1 + bq) * parent_1[j]) + (1 - bq) * parent_2[j])
                child_2 = 0.5 * (((1 - bq) * parent_1[j]) + (1 + bq) * parent_2[j])
                child[p, j] = Bounds(child_1, Problem['LB'][j], Problem['UB'][j])
                if p + 1 < N:
                    child[p + 1, j] = Bounds(child_2, Problem['LB'][j], Problem['UB'][j])
        else:
            child[p] = parent_1
            if p + 1 < N:
                child[p + 1] = parent_2
        for j in range(V):
            if np.random.rand() < pm:
                delta = (2 * np.random.rand()) ** (1 / (mum + 1)) - 1 if np.random.rand() < 0.5 else 1 - (
                            2 * (1 - np.random.rand())) ** (1 / (mum + 1))
                child[p, j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
                child[p, j] = Bounds(child[p, j], Problem['LB'][j], Problem['UB'][j])
            if p + 1 < N and np.random.rand() < pm:
                delta = (2 * np.random.rand()) ** (1 / (mum + 1)) - 1 if np.random.rand() < 0.5 else 1 - (
                            2 * (1 - np.random.rand())) ** (1 / (mum + 1))
                child[p + 1, j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
                child[p + 1, j] = Bounds(child[p + 1, j], Problem['LB'][j], Problem['UB'][j])
        p += 2
    P = {'x': child, 'f': parent_chromosome['f']}  # ensure 'f' is included
    return P


import time
import numpy as np
import matplotlib.pyplot as plt

def rGA(Problem, InitialPopulation, Options, *args):
    DefaultOpt = {'MaxObj': 2000, 'MaxGen': 200, 'PopSize': 40, 'EliteProp': 0.1,
                  'TourSize': 2, 'Pcross': 0.9, 'Icross': 20, 'Pmut': 0.1, 'Imut': 20,
                  'CPTolerance': 1.0e-6, 'CPGenTest': 0.01, 'Verbosity': 0}
    if isinstance(Problem, str) and Problem.lower() == 'defaults':
        return DefaultOpt

    Options = {**DefaultOpt, **(Options or {})}

    MaxGenerations = Options['MaxGen']
    MaxEvals = Options['MaxObj']
    Pop = Options['PopSize']
    Elite = Options['EliteProp']
    Tour = Options['TourSize']
    Pc = Options['Pcross']
    Ic = Options['Icross']
    Pm = Options['Pmut']
    Im = Options['Imut']

    Problem['Stats'] = {'ObjFunCounter': 0, 'GenCounter': 0, 'Best': [], 'Worst': [], 'Mean': [], 'Std': []}
    Problem['Tolerance'] = Options['CPTolerance']
    Problem['GenTest'] = Options['CPGenTest']
    Problem['Verbose'] = Options['Verbosity']

    Problem, Population = InitPopulation(Problem, InitialPopulation, Pop, *args)

    def update_stats(Population):
        best = np.min(Population['f'])
        worst = np.max(Population['f'])
        mean = np.mean(Population['f'])
        std = np.std(Population['f'])
        Problem['Stats']['Best'].append(best)
        Problem['Stats']['Worst'].append(worst)
        Problem['Stats']['Mean'].append(mean)
        Problem['Stats']['Std'].append(std)

    update_stats(Population)
    if Problem['Verbose']:
        print('rGA is alive... ')

    while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
        if Problem['Stats']['GenCounter'] > 0 and not Problem['Stats']['GenCounter'] % int(
                Problem['GenTest'] * MaxGenerations):
            if abs(Problem['Stats']['Best'][-1] - Problem['Stats']['Best'][-int(Problem['GenTest'] * MaxGenerations)]) < \
                    Problem['Tolerance']:
                print(
                    'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest generations')
                break

    # while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
    #     # Stop if the improvement is inferior to the Tolerance in the last generations
    #     if (Problem['Stats']['GenCounter'] > 0 and
    #             Problem['Stats']['GenCounter'] % math.ceil(Problem['GenTest'] * MaxGenerations) == 0):
    #
    #         current_best = Problem['Stats']['Best'][Problem['Stats']['GenCounter']]
    #         previous_best = Problem['Stats']['Best'][
    #             Problem['Stats']['GenCounter'] - math.ceil(Problem['GenTest'] * MaxGenerations)]
    #
    #         if abs(current_best - previous_best) < Problem['Tolerance']:
    #             print(
    #                 'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest generations')
    #             break

        Problem['Stats']['GenCounter'] += 1
        elitesize = int(Pop * Elite)
        pool = Pop - elitesize
        parent_chromosome = tournament_selection(Population, pool, Tour)
        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)
        Population['x'][elitesize:] = offspring_chromosome['x'][:pool]
        for i in range(elitesize, Pop):
            Problem, Population['f'][i] = ObjEval(Problem, Population['x'][i], *args)
        sorted_indices = np.argsort(Population['f'])
        Population['x'] = Population['x'][sorted_indices]
        Population['f'] = Population['f'][sorted_indices]
        update_stats(Population)

        if Problem['Verbose'] == 2 and Problem['Stats']['GenCounter'] % 10 == 0:  # Update plot every 10 generations
            def plot_with_backoff():
                retries = 5
                delay = 1
                for attempt in range(retries):
                    try:
                        plt.ion()
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        xx, yy = np.meshgrid(np.linspace(Problem['LB'][0], Problem['UB'][0], 80),
                                             np.linspace(Problem['LB'][1], Problem['UB'][1], 80))
                        zz = np.array(
                            [[Problem['ObjFunction'](np.array([xx[i, j], yy[i, j]]), *args) for j in range(80)] for i in range(80)])
                        ax1.set_title('Objective function')
                        surf = ax1.contourf(xx, yy, zz, 20)
                        fig.colorbar(surf, ax=ax1)
                        ax1.set_xlabel('x')
                        ax1.set_ylabel('y')
                        ax2.set_title(f'Population at generation: {Problem["Stats"]["GenCounter"]}')
                        ax2.set_xlabel('x')
                        ax2.set_ylabel('y')
                        scatter = ax2.scatter(Population['x'][:, 0], Population['x'][:, 1], c='blue')
                        scatter.set_offsets(Population['x'][:, :2])
                        plt.draw()
                        plt.pause(0.1)
                        return
                    except Exception as e:
                        print(f"Plot generation failed (attempt {attempt + 1}/{retries}): {e}")
                        time.sleep(delay)
                        delay *= 2
                raise Exception("Failed to generate plot after several attempts")

            plot_with_backoff()

    if Problem['Verbose']:
        plt.ioff()
        if Problem['Verbose'] == 2:
            # Generate the final plot with the best solution
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            xx, yy = np.meshgrid(np.linspace(Problem['LB'][0], Problem['UB'][0], 80),
                                 np.linspace(Problem['LB'][1], Problem['UB'][1], 80))
            zz = np.array(
                [[Problem['ObjFunction'](np.array([xx[i, j], yy[i, j]]), *args) for j in range(80)] for i in range(80)])
            ax1.set_title('Objective function')
            surf = ax1.contourf(xx, yy, zz, 20)
            fig.colorbar(surf, ax=ax1)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax2.set_title(f'Final Population at generation: {Problem["Stats"]["GenCounter"]}')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            scatter = ax2.scatter(Population['x'][:, 0], Population['x'][:, 1], c='blue')
            scatter.set_offsets(Population['x'][:, :2])
            ax2.scatter(Population['x'][0, 0], Population['x'][0, 1], c='red', label='Best Chromosome')
            plt.show()
        print('Maximum number of iterations or objective function evaluations reached')

    BestChrom = Population['x'][0]
    BestChromObj = Population['f'][0]
    RunData = Problem['Stats']
    return BestChrom, BestChromObj, RunData



if __name__ == "__main__":

    # Define an example objective function
    def sphere_function(x):
        return np.sum(x ** 2)


    # Example usage
    if __name__ == "__main__":
        Problem = {
            'Variables': 2,
            'ObjFunction': sphere_function,
            'LB': np.array([-15, -15]),
            'UB': np.array([15, 15])
        }
        InitialPopulation = [{'x': np.array([-2, 3])}]
        Options = {
            'Pmut': 1 / Problem['Variables'],
            'MaxGen': 20000,
            'Verbosity': 1
        }
        BestChrom, BestChromObj, RunData = rGA(Problem, InitialPopulation, Options)
        print(f"Best Chromosome: {BestChrom}")
        print(f"Best Chromosome Objective Value: {BestChromObj}")
        print(f"RunData: {RunData}")

# Example usage
