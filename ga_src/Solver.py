from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination
from .Problem import CVRP, LS
from sklearn.metrics import pairwise_distances
import numpy as np

from .local_search import local_search


def random_sample(n_samples, n_var):
    X = np.full((n_samples, n_var), 0, dtype=int)
    for i in range(n_samples):
        X[i, :] = np.random.permutation(n_var)
    return X

def solve(data, run_id, LS_LST=None, PLS=0.01, init_sols=None, pop_size=100, max_gen=20):
    dist_map = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    capacity = int(data['CAPACITY'])
    demand = np.reshape(data['DEMANDS'], -1).tolist()


    problem = CVRP(len(demand)-1, capacity, demand, dist_map, LS_LST, PLS=PLS)
    X = random_sample(n_samples=pop_size, n_var=problem.n_var)
    if init_sols is not None and len(init_sols) > 0:
        for idx, init_sol in enumerate(init_sols):
            # modified, cost, _ = local_search(init_sol, dist_map, capacity, demand, np.inf, optlst=LS_LST)
            # modified = init_sol
            X[idx, :] = problem._encoding(init_sol)

    # apply local search for the first few solutions


    algorithm = GA(
        pop_size=pop_size,
        sampling=X,
        mutation=InversionMutation(),
        crossover=OrderCrossover(),
        repair=LS(),
        eliminate_duplicates=True
    )

    # if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
    termination = DefaultSingleObjectiveTermination(period=np.inf, n_max_gen=max_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=run_id,
        save_history=True,
        verbose=False
    )
    return res