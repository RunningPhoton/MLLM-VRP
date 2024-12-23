import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair

from verypy.util import objf as route_cost
from .local_search import local_search



class LS(Repair):
    def _do(self, problem, X, **kwargs):
        # customized init solution
        # if problem.init_sol is not None:
        #     if problem.first == True:
        #         problem.first = False
        #         modified, cost, _ = local_search(problem.init_sol, problem.dist_map, problem.capacity, problem.demand, np.inf, optlst=problem.LS_LST)
        #         X[0] = problem._encoding(modified)
        # perform local search for the entire population
        for k in range(len(X)):
            x = X[k]
            # sol = problem._decoding(x)
            if np.random.rand() < problem.PLS:
                sol = problem._decoding(x)
                sol, cost, _ = local_search(sol, problem.dist_map, problem.capacity, problem.demand, np.inf, optlst=problem.LS_LST)
                X[k] = problem._encoding(sol)
        return X



# define the custom problem class for CVRP
class CVRP(ElementwiseProblem):

    # initialize the problem with the data
    def __init__(self, n_customers, capacity, demand, dist_map, LS_LST, PLS=0.05, init_sol=None, **kwargs):
        # number of variables is equal to number of customers
        n_var = n_customers
        # number of objectives is: minimize distance
        n_obj = 1
        # lower bound for each variable is 1
        xl = 0
        # upper bound for each variable is n_customer + 1 (not included)
        xu = n_customers
        # call the super class constructor
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=int, **kwargs)
        # store the data as attributes
        self.n_customers = n_customers
        self.capacity = capacity
        self.demand = demand
        self.dist_map = dist_map
        self.LS_LST = LS_LST
        self.PLS = PLS
        # self.init_sol = init_sol
        # self.first = True if init_sol is not None else False

    def _decoding(self, x):
        return self._dp(x)
        # sol = [0]
        # cur_demand = 0
        # for vid in x:
        #     val = vid + 1
        #     if cur_demand + self.demand[val] <= self.capacity:
        #         cur_demand += self.demand[val]
        #         sol.append(val)
        #     else:
        #         cur_demand = self.demand[val]
        #         sol.append(0)
        #         sol.append(val)
        # sol.append(0)
        # # check
        # return sol
    def _dp(self, x):
        # initialize the DP table
        XID = (x + 1).tolist()
        n = len(XID)  # number of customers
        dp = [float('inf')] * (n + 1) # the least cost of serving customers
        prev = [None] * (n + 1)
        dp[0], dp[1] = 0, self.dist_map[0, XID[0]] * 2
        sum_d = [0] * (n + 1)  # 服务前i个点，需求和
        dist = [0] * (n + 1)  # 服务前i个点，距离和
        for i, cid in enumerate(XID):
            sum_d[i+1] = sum_d[i] + self.demand[cid]
        for i in range(2, n + 1):
            pc, nc = XID[i-2], XID[i-1] # travel from previous node to new node
            dist[i] = dist[i-1] + self.dist_map[pc, nc]

        for i in range(2, n + 1):
            cid = XID[i-1]
            for j in range(i):
                # 服务j+1 ~ i
                t_demand = sum_d[i] - sum_d[j]
                if t_demand > self.capacity + 1e-3: continue
                pid = XID[j]
                t_cost = dist[i] - dist[j+1] + dp[j] + self.dist_map[0, pid] + self.dist_map[cid, 0]
                if t_cost < dp[i]:
                    dp[i] = t_cost
                    prev[i] = j

        sol = XID
        while n is not None:
            sol.insert(n, 0)
            n = prev[n]
        return sol
    def _encoding(self, sol):
        x = []
        for val in sol:
            if val != 0:
                x.append(val-1)
        return x
    # define the evaluation function
    def _evaluate(self, x, out, *args, **kwargs):
        sol = self._decoding(x)
        out['F'] = route_cost(sol, self.dist_map)
