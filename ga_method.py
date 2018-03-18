import random
import math
import copy
import functools
from timeit import default_timer as timer

#paper: http://old.mii.lt/files/mii_dis_2014_vaira.pdf


def dist(s, e):
    return math.sqrt((s.x - e.x) ** 2 + (s.y - e.y) ** 2)

class Solution(object):
    def __init__(self, cities, max_routes = 5, capacity = 100):
        self.routes = []
        self.unsolved = []
        self.nodes = cities[1:]
        self.start = cities[0]
        self.max_routes = max_routes
        self.capacity = capacity
        
    def satisfied(self, arc, node, route):
        current = sum([x.capacity for x in route])
        return current + node.capacity <= self.capacity
    
    def make_arcs(self, r, i = 0):
        if len(r) == 0: return []
        return list(zip([self.start] + r, r + [self.start], [i]*(len(r) + 1)))

    def cost(self, arc, ins):
        return dist(arc[0], ins) + dist(ins, arc[1])
        
    def score(self):
        score = 0
        for r in self.routes:
            k = list(zip([self.start] + r, r + [self.start]))
            score = score + sum(map(lambda x: dist(x[0], x[1]), k))
        return score
    
    def insert(self, rs_orig, nodes_orig):
        unsolved = []
        nodes = copy.deepcopy(nodes_orig)
        rs = copy.deepcopy(rs_orig)
        while nodes:
            i = random.randint(0, len(nodes) - 1)
            n = nodes.pop(i)
            new_arc = []
            for i in range(len(rs)):
                feasible = [x for x in self.make_arcs(rs[i], i) if self.satisfied(x, n, rs[i])]
                new_arc.extend(feasible)
                
            if len(new_arc) == 0 and len(rs) < self.max_routes and n.capacity <= self.capacity:
                rs.append([n])
            else:
                if len(new_arc) != 0:
                    arc = min(new_arc, key=lambda x: self.cost(x, n))
                    try:
                        new_idx = rs[arc[2]].index(arc[1])
                    except:
                        new_idx = len(rs[arc[2]])
                    rs[arc[2]].insert(new_idx, n)
                else:
                    unsolved.append(n)
        return rs, unsolved
        
    def create(self):
        self.routes, self.unsolved = self.insert(self.routes, self.nodes)

    def print_sol(self):
        for i in range(len(self.routes)):
            cost = [x.capacity for x in self.routes[i]]
            print "Vehicle %d (%d/%d): %s" % (i + 1, sum(cost), self.capacity, str([x.city_id for x in self.routes[i]]))
        print "Total cost: %.3f" % self.score()
        if len(self.unsolved) != 0:
            print "Warning: this solution has unresolved cities: %s" % (str([x.city_id for x in self.unsolved]))
        
    def greedy_construct(self):
        snodes = sorted([x for x in self.nodes if x.capacity <= self.capacity], key=lambda x: dist(self.start, x))
        route_idx = 0
        
        while self.nodes:
            current_route = [] if len(self.routes) <= route_idx else self.routes[route_idx]
            remains = [k for k in self.nodes if self.satisfied([], k, current_route)]            
            if (len(current_route) == 0 or len(remains) == 0) and len(self.routes) < self.max_routes and snodes:
                fst = snodes.pop(0)
                self.routes.append([fst])
                self.nodes.remove(fst)
                if len(remains) == 0:
                    route_idx = route_idx + 1
            else:
                if len(remains) != 0:
                    m = self.routes[route_idx][-1]
                    remains_sorted = sorted(remains, key=lambda x: dist(m, x))
                    item = remains_sorted[0]
                    self.routes[route_idx].append(item)
                    self.nodes.remove(item)
                    snodes.remove(item)
                else:
                    self.unsolved.extend(self.nodes)
                    self.nodes = []


def len_lists(ls):
    return sum([len(l) for l in ls])

def index2d(ls, idx):
    if len(ls) == 0 or len_lists(ls) <= idx:
        raise Exception("Index out of range")
    sz = [len_lists(ls[:i+1]) for i in range(len(ls))]
    for i in range(len(sz)):
        if idx < sz[i]:
            prev = 0 if i == 0 else sz[i - 1]
            return (i, idx - prev)
        
def compare_sol(s, os):
    if len(s.unsolved) != len(os.unsolved):
        return len(os.unsolved) - len(s.unsolved)
    if len(s.routes) != len(os.routes):
        return len(os.routes) - len(s.routes)
    return os.score() - s.score()

class Ga(object):
    def __init__(self, tour, init_pop=10, iterations=50, time=60):
        self.init_pop = init_pop
        self.iterations = iterations
        self.time = time
        self.population = []
        self.tour = tour
        self.ps1 = init_pop
        self.pl1 = self.ps1 / 5
        self.mp = 0.1
        self.ps2 = self.ps1 / 10
        self.pl2 = self.ps2 / 5
        self.ipop2 = iterations / 10
        self.stats = []
        random.seed()
        
    def init(self):
        for i in range(self.init_pop):
            sol = Solution(self.tour.destinationCities, self.tour.num_cars, self.tour.car_limit)
            sol.create()
            self.population.append(sol)
            
    def choose(self, ranked):
        c = copy.deepcopy(ranked)
        choices = []
        for i in range(2):
            m = range(1, len(c) + 1)
            m = [sum(m[:i+1]) for i in range(len(m))]
            m = map(lambda x: x * 1.0 / m[-1], m)

            num = random.random()
            j = 0
            while num > m[j]:
                j = j + 1
            choices.append(c.pop(j))
        return choices

    def run_auxiliary(self, aux):
        runiter = 0
        last_best = None
        while runiter < self.ipop2:
            aux = sorted(aux, key=functools.cmp_to_key(compare_sol))
            aux = aux[len(aux) - self.ps2:]
            if self.not_improved(last_best, aux[-1]):
                runiter = runiter + 1
            else:
                runiter = 0
            last_best = aux[-1]
            for i in range(self.pl2):
                mom, dad = self.choose(aux)
                fst = self.crossover(mom, dad)
                snd = self.crossover(dad, mom)
                aux.extend([fst, snd])
                if random.random() < self.mp:
                    routes, nodes = self.mutation(fst)
                    s = Solution(self.tour.destinationCities, self.tour.num_cars, self.tour.car_limit)
                    s.routes, s.unsolved = s.insert(routes, nodes)
                    aux.append(s)
        return aux

    def best_auxiliary(self, routes, nodes):
        aux_pop = []
        for i in range(self.ps2):
            s = Solution(self.tour.destinationCities, self.tour.num_cars, self.tour.car_limit)
            s.routes, s.unsolved = s.insert(routes, nodes)
            aux_pop.append(s)
        aux_pop = self.run_auxiliary(aux_pop)
        aux_pop = sorted(aux_pop, key=functools.cmp_to_key(compare_sol))
        return aux_pop[-1]

    def aver_best(self, pop):
        best = pop[-1].score()
        aver = sum([p.score() for p in pop]) / len(pop)
        return (best, aver)
    
    def not_improved(self, last, cur):
        if not last:
            return False
        return compare_sol(last, cur) >= 0
    
    def run(self):        
        runiter = 0
        self.stats = []
        start = timer()
        elapsed = 0
        last_best = None
        while runiter < self.iterations and elapsed < self.time:
            pop = sorted(self.population, key=functools.cmp_to_key(compare_sol))
            self.population = pop[len(pop) - self.ps1:]
            self.stats.append(self.aver_best(self.population))
            if self.not_improved(last_best, self.population[-1]):
                runiter = runiter + 1
            else:
                runiter = 0
            last_best = self.population[-1]
            for i in range(self.pl1):
                mom, dad = self.choose(self.population)
                fst = self.crossover(mom, dad)
                snd = self.crossover(dad, mom)
                self.population.extend([fst, snd])
                if random.random() < self.mp:
                    routes, nodes = self.mutation(fst)
                    aux = self.best_auxiliary(routes, nodes)
                    self.population.append(aux)
            elapsed = timer() - start
        pop = sorted(self.population, key=functools.cmp_to_key(compare_sol))
        elapsed = timer() - start
        print "Takes: %.3f seconds" % elapsed
        return pop[-1]
        
    def test(self):
        self.init()
        sol = self.run()
        print "GA method:"
        sol.print_sol()
                
    def mutation(self, orig):
        if random.random() <= 0.33:
            return self.simple_mutation(orig)
        if random.random() <= 0.66:
            return self.cluster_mutation(orig)
        return self.routes_mutation(orig)
    
    def crossover(self, mom, dad):
        if random.random() <= 0.5:
            return self.common_arc_crossover(mom, dad)
        return self.common_node_crossover(mom, dad)
    
    def same_arc(self, a, other):
        return a[0].same_city(other[0]) and a[1].same_city(other[1])

    def get_elem(self, routes, idx):
        tidx = index2d(routes, idx)
        n = routes[tidx[0]].pop(tidx[1])
        return n

    def simple_mutation(self, orig):
        num = int(random.random() * 0.5 * len(self.tour.destinationCities))
        routes = copy.deepcopy(orig.routes)
        nodes = copy.deepcopy(orig.unsolved)
        num = num - len(nodes)
        for i in range(num):
            idx = random.randint(0, len_lists(routes) - 1)
            nodes.append(self.get_elem(routes, idx))
        routes = filter(lambda x: len(x) > 0, routes)
        return routes, nodes
    
    def cluster_mutation(self, orig):
        num = int(random.random() * 0.5 * len(self.tour.destinationCities))
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        num = num - len(unsolved)
        idx = random.randint(0, len_lists(routes) - 1)
        n = self.get_elem(routes, idx)
        unsolved.append(n)
        flat = [item for r in routes for item in r]
        flat_idx = list(zip(flat, range(len(flat))))
        flat_idx = sorted(flat_idx, key=lambda x: dist(x[0], n))
        to_del = []
        for i in range(num):
            node = flat_idx.pop(0)
            to_del.append(node[1])
        for i in sorted(to_del, reverse=True):
            node = self.get_elem(routes, i)
            unsolved.append(node)
        routes = filter(lambda x: len(x) > 0, routes)            
        return routes, unsolved
    
    def routes_mutation(self, orig):
        num = int(random.random() * 0.5 * len(orig.routes))
        routes = copy.deepcopy(orig.routes)
        unsolved = copy.deepcopy(orig.unsolved)
        for i in range(num):
            idx = random.randint(0, len(routes) - 1)
            r = routes.pop(idx)
            unsolved.extend(r)
        routes = filter(lambda x: len(x) > 0, routes)            
        return routes, unsolved
            
    def common_arc_crossover(self, mom, dad):
        s = Solution(self.tour.destinationCities, self.tour.num_cars, self.tour.car_limit)
        unsolved = copy.deepcopy(mom.unsolved)
        dad_arcs = []
        mom_arcs = []
        both_arcs = [[] for i in range(len(mom.routes))]
        for r in dad.routes:
            dad_arcs.extend(dad.make_arcs(r))
        for i in range(len(mom.routes)):
            arcs = mom.make_arcs(mom.routes[i], i)
            mom_arcs.extend(arcs)
            
        for a in mom_arcs:
            if any(map(lambda x: self.same_arc(a, x), dad_arcs)):
                both_arcs[a[2]].append(a)
            else:
                if a[0].city_id != 0:
                    unsolved.append(a[0])
        
        both_arcs = filter(lambda x: len(x) > 0, both_arcs)
        routes = []
        for r in both_arcs:
            part = []
            for arc in r:
                added_nodes = [x for x in arc[:-1] if x not in part and x.city_id != 0]
                part.extend(added_nodes)
                for p in part:
                    if p in unsolved:
                        unsolved.remove(p)
            if len(part) > 0:
                routes.append(part)
        s.routes, s.unsolved = s.insert(routes, unsolved)
        return s
            
    def common_node_crossover(self, mom, dad):
        s = Solution(self.tour.destinationCities, self.tour.num_cars, self.tour.car_limit)
        unsolved = copy.deepcopy(mom.unsolved)
        routes = []
        for r in dad.routes:
            rtemp = []
            route_idx = []
            for n in r:
                idx = -1
                for i in range(len(mom.routes)):
                    if n in mom.routes[i]:
                        idx = i
                if idx < 0:
                    if n not in unsolved:
                        unsolved.append(n)
                else:
                    if idx in route_idx:
                        rtemp[route_idx.index(idx)].append(n)
                    else:
                        rtemp.append([n])
                        route_idx.append(idx)
                    if n in unsolved:
                        unsolved.remove(n)
            if len(rtemp) == 0:
                continue
            rtemp = sorted(rtemp, key=len, reverse=True)
            best = rtemp.pop(0)
            routes.append(best)
            unsolved.extend([nodei for temp in rtemp for nodei in temp])
        routes = filter(lambda x: len(x) > 0, routes)
        s.routes, s.unsolved = s.insert(routes, unsolved)
        return s
        
def ga_sol(tour, num_start, num_iter, time=180):
    ga = Ga(tour, num_start, num_iter, time)
    ga.test()
    return ga.stats
    
def greedy_sol(tour):
    s = Solution(tour.destinationCities, tour.num_cars, tour.car_limit)
    s.greedy_construct()
    print "Greedy solution:"
    s.print_sol()
    return s

