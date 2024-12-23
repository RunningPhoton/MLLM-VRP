import time

import numpy as np
from verypy.local_search.solution_operators import do_3optstar_move
from verypy.local_search.inter_route_operators import do_2optstar_move, do_1point_move, do_2point_move, do_insert_move, do_redistribute_move, do_chain_move
from verypy.local_search.intra_route_operators import do_2opt_move, do_3opt_move, do_relocate_move, do_exchange_move
from verypy.local_search import LSOPT
from verypy.routedata import RouteData
# Helpers, so simple that they are sure to work right
from verypy.util import routes2sol, sol2routes
from verypy.util import objf as route_l
from verypy.util import totald as route_d
from ga_src.env import MAX_LS_LEN, MAX_LS_ITER, INNER_LS_ITER
from sklearn.metrics import pairwise_distances

'''

* routeZ is a RouteData object, the data members are
    1. route as a list of node indices. Must begin and end to depot (0).
    2. current route cost
    3. current route demand, can be None is C = None
* D is the numpy-compatibe distance matrix as with intra route heuristics.
* C is the optional capacity constraint (can be None). If set, also give d.
* d is a list of customer demands, and of the depot it is d[0]=0,
    The parameter must be set if  C is given.
* L is the maximun route length/duration/cost constraint

* route is a list of nodes forming a route.The route must start and end to the depot (index 0), e.g. [0,1,2,3,0].
* D is the symmetric full distance matrix, given as numpy-like 2darray. Thus, D[i,j] gives the distance between nodes i and j.
* strategy is either FIRST_ACCEPT (default) or BEST_ACCEPT. First accept returns a modified route as soon as the first improvement is encoutered. Best accept tries all possible combinations and returns the best one.
'''

def rlst2sol(rlst):
    solution = [0]
    for seq in rlst:
        solution = solution + seq + [0]
    print(solution)
    return solution

def sol2rlst(solution):
    rlst = []
    temp = []
    for val in solution[1:]:
        if val != 0:
            temp.append(val)
        else:
            rlst.append(temp)
            temp = []
    print(rlst)
    return rlst

def _normalise_route_order(sol):
    routes = sol2routes(sol)
    # make sure route nodes are directed so that smaller end point is first
    for route in routes:
        if route[1]>route[-2]:
            route.reverse()
    # make sure routes are from the smallest first node first
    routes.sort()
    return routes2sol(routes)

def LS_all_move(solution, D, C, d, L, movefun, max_iter=INNER_LS_ITER):
    # solution = rlst2sol(rlst)
    # improve the quality via moving MAX_ITER times
    for _ in range(max_iter):
        new_sol, new_delta = movefun(solution=solution, D=D, C=C, demands=d, L=L)
        if new_sol is not None:
            solution = new_sol
        else:
            return None
    return solution

def LS_2route_move(solution, D, C, d, L, movefun, max_iter=INNER_LS_ITER):
    # solution = rlst2sol(rlst)
    # improve the quality via moving MAX_ITER times
    routes = [RouteData(r, route_l(r, D), route_d(r, d), None) for r in sol2routes(solution)]
    for _ in range(max_iter):
        improve = False
        for _r1, r1rd in enumerate(routes):
            for _r2, r2rd in enumerate(routes):
                if _r2 == _r1: continue
                nr1rd, nr2rd, best_delta = movefun(r1rd, r2rd, D, C=C, d=d, L=L)
                if best_delta is not None:
                    improve = True
                    routes[_r1] = nr1rd
                    routes[_r2] = nr2rd
                    break
            if improve == True: break
        if improve == False:
            return None
    new_routes = [route[0] for route in routes]
    return routes2sol(new_routes)

def LS_1route_move(solution, D, C, d, L, movefun, max_iter=INNER_LS_ITER):
    routes = sol2routes(solution)
    for _ in range(max_iter):
        improve = False
        for _r, rd in enumerate(routes):
            nrd, best_delta = movefun(rd, D)
            if best_delta is not None:
                improve = True
                routes[_r] = nrd
                break
        if improve == False:
            return None
    return routes2sol(routes)


# do_2optstar_move, do_1point_move, do_2point_move, do_insert_move, do_redistribute_move, do_chain_move
# from verypy.local_search.intra_route_operators import do_2opt_move, do_3opt_move, do_relocate_move, do_exchange_move
# do_2opt_move(route, D, , best_delta=None):
def LS_impl(solution, D, C, d, L, optlst, iterations):
    # solution = rlst2sol(rlst)
    functions = {
        # exchange customers among all routes
        0: do_3optstar_move,
        # exchange customers between two routes
        1: do_2optstar_move,
        2: do_1point_move,
        3: do_2point_move,
        4: do_insert_move,
        5: do_redistribute_move,
        # exchange customers within one route
        6: do_2opt_move,
        7: do_3opt_move,
        8: do_relocate_move,
        9: do_exchange_move,
    }
    for _ in range(iterations):
        improve = False
        for lid in optlst:
            assert lid >= 0 and lid < 10, 'local search operation errors'
            if lid == 0:
                new_solution = LS_all_move(solution, D, C, d, L, movefun=functions[lid])
            elif lid > 0 and lid < 6:
                new_solution = LS_2route_move(solution, D, C, d, L, movefun=functions[lid])
            else:
                new_solution = LS_1route_move(solution, D, C, d, L, movefun=functions[lid])
            if new_solution is not None:
                solution = new_solution
                improve = True
        if improve == False:
            break
    # return sol2rlst(solution)
    return solution
def local_search(solution, D, C, d, L, optlst=None, max_length=MAX_LS_LEN, iterations=MAX_LS_ITER):
    solution = _normalise_route_order(solution)
    # D = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    # C = int(data['CAPACITY'])
    # d = np.reshape(data['DEMANDS'], -1)
    # L = np.inf
    if optlst == None:
        optlst = [2, 3, 6, 7]
    optlst = optlst[:max_length]
    tm_slot = time.time()
    new_solution = LS_impl(solution, D, C, d, L, optlst, iterations=iterations)
    travel_cost = route_l(new_solution, D)
    run_tm = time.time()-tm_slot
    return _normalise_route_order(new_solution), travel_cost, run_tm
