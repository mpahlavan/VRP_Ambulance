"""
Genetic Algorithm Solver for PVRP
Implements GA-based optimization for Perishable Vehicle Routing Problem
"""

import random
import math
from typing import List, Optional, Tuple
from multiprocessing import Pool
from marpdan.dep import tqdm

from ._ga_operators import (
    Chromosome,
    routes_to_chromosome,
    tournament_selection,
    order_crossover,
    pmx_crossover,
    two_opt_mutation,
    swap_mutation,
    insert_mutation,
    inversion_mutation,
    or_opt_mutation,
)


class PVRP_GeneticAlgorithm:
    """
    Genetic Algorithm solver for Perishable Vehicle Routing Problem

    Attributes:
        nodes: Tensor of shape (N+1, 4) with [x, y, demand, spoilage_time]
        veh_count: Number of vehicles
        veh_capa: Vehicle capacity
        veh_speed: Vehicle speed
    """

    # Cost weights for GA internal fitness (scaled up for better search)
    # Relative proportions based on environment coefficients:
    #   spoilage_penalty=1, unserved_penalty=2, dist_penalty_coef=0.05,
    #   idle_penalty_coef=0.5, additional_late_penalty=1
    # Note: Final comparison uses PVRP environment's cost function
    DIST_WEIGHT = 1.0           # Aligned with dist_penalty_coef * 20 scale
    LATE_PICKUP_WEIGHT = 20.0   # Aligned with spoilage_penalty * 20 scale
    UNSERVED_WEIGHT = 40.0      # Aligned with unserved_penalty=2 * 20 scale
    IDLE_VEH_WEIGHT = 10.0      # Aligned with idle_penalty_coef=0.5 * 20 scale
    LATE_DEPOT_WEIGHT = 20.0    # Aligned with additional_late_penalty * 20 scale
    CAPACITY_VIOLATION_WEIGHT = 100.0  # High penalty for exceeding vehicle capacity

    def __init__(
        self,
        nodes,
        veh_count: int,
        veh_capa: int,
        veh_speed: float,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.20,
        elite_count: int = 5,
        tournament_size: int = 5,
        early_stop_generations: int = 50,
    ):
        """
        Initialize GA solver
        """
        # Convert to list for easier manipulation
        self.nodes = [[float(nodes[i, j]) for j in range(nodes.shape[1])]
                      for i in range(nodes.shape[0])]
        self.n_customers = len(self.nodes) - 1  # Exclude depot
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed

        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.early_stop_generations = early_stop_generations

        # Precompute distances and feasibility
        self._precompute_distances()
        self._identify_infeasible_nodes()

    def _precompute_distances(self):
        """Precompute distance matrix between all nodes"""
        n = len(self.nodes)
        self.distances = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dx = self.nodes[i][0] - self.nodes[j][0]
                dy = self.nodes[i][1] - self.nodes[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                self.distances[i][j] = dist
                self.distances[j][i] = dist

    def _identify_infeasible_nodes(self):
        """Identify nodes that cannot be served due to spoilage constraints"""
        self.infeasible_nodes = set()

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            spoilage_time = node[3]

            # Time to reach node from depot
            travel_to = self.distances[0][i] / self.veh_speed
            # Time to return to depot
            travel_back = self.distances[i][0] / self.veh_speed

            # Latest pickup time = spoilage_time - return_time
            latest_pickup = spoilage_time - travel_back

            # If we can't reach in time, mark as infeasible
            if travel_to > latest_pickup:
                self.infeasible_nodes.add(i)

        self.feasible_customers = [i for i in range(1, len(self.nodes))
                                   if i not in self.infeasible_nodes]

    def _compute_route_time(self, route: List[int]) -> float:
        """Compute total time to complete a route"""
        if not route:
            return 0.0

        time = 0.0
        pos = 0  # Start at depot
        for customer in route:
            time += self.distances[pos][customer] / self.veh_speed
            pos = customer
        # Return to depot
        time += self.distances[pos][0] / self.veh_speed
        return time

    def _evaluate_fitness(self, chromo: Chromosome) -> float:
        """
        Evaluate chromosome fitness (cost to minimize)
        """
        routes = chromo.decode_routes(self.veh_count)

        total_distance = 0.0
        late_pickups = 0
        late_at_depot = 0
        idle_vehicles = 0
        capacity_violations = 0  # Track capacity constraint violations
        served_customers = set()

        for route in routes:
            if not route:
                idle_vehicles += 1
                continue

            current_pos = 0  # Start at depot
            current_time = 0.0
            route_customers = []
            route_load = 0  # Track load for this route

            for customer in route:
                if customer in self.infeasible_nodes:
                    continue
                if customer in served_customers:
                    continue  # Skip duplicates

                # Check capacity constraint BEFORE serving
                route_load += 1
                if route_load > self.veh_capa:
                    capacity_violations += 1  # Penalize each customer over capacity

                # Travel to customer
                travel_time = self.distances[current_pos][customer] / self.veh_speed
                current_time += travel_time
                total_distance += self.distances[current_pos][customer]

                served_customers.add(customer)
                route_customers.append(customer)

                # Check spoilage constraint
                spoilage_time = self.nodes[customer][3]
                return_time = self.distances[customer][0] / self.veh_speed
                latest_pickup = spoilage_time - return_time

                if current_time > latest_pickup:
                    late_pickups += 1

                current_pos = customer

            # Return to depot
            if route:
                total_distance += self.distances[current_pos][0]
                depot_arrival = current_time + self.distances[current_pos][0] / self.veh_speed

                # Check if any item spoils on way back to depot
                for customer in route_customers:
                    spoilage_time = self.nodes[customer][3]
                    if depot_arrival > spoilage_time:
                        late_at_depot += 1

        # Count unserved feasible customers
        unserved = len(self.feasible_customers) - len(served_customers & set(self.feasible_customers))

        # Calculate total cost
        cost = (
            self.DIST_WEIGHT * total_distance +
            self.LATE_PICKUP_WEIGHT * late_pickups +
            self.UNSERVED_WEIGHT * unserved +
            self.IDLE_VEH_WEIGHT * idle_vehicles +
            self.LATE_DEPOT_WEIGHT * late_at_depot +
            self.CAPACITY_VIOLATION_WEIGHT * capacity_violations
        )

        chromo.fitness = cost
        return cost

    def _create_greedy_solution(self) -> Chromosome:
        """Create solution using nearest neighbor heuristic"""
        routes = [[] for _ in range(self.veh_count)]
        unvisited = set(self.feasible_customers)
        loads = [0] * self.veh_count
        times = [0.0] * self.veh_count
        positions = [0] * self.veh_count

        while unvisited:
            best_insertion = None
            best_cost = float('inf')

            for veh in range(self.veh_count):
                if loads[veh] >= self.veh_capa:
                    continue

                for customer in unvisited:
                    # Check feasibility
                    travel_time = self.distances[positions[veh]][customer] / self.veh_speed
                    arrival_time = times[veh] + travel_time

                    spoilage_time = self.nodes[customer][3]
                    return_time = self.distances[customer][0] / self.veh_speed
                    latest_pickup = spoilage_time - return_time

                    if arrival_time <= latest_pickup:
                        # Use distance as greedy cost
                        cost = self.distances[positions[veh]][customer]
                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (veh, customer, arrival_time)

            if best_insertion is None:
                break

            veh, customer, arrival_time = best_insertion
            routes[veh].append(customer)
            loads[veh] += 1
            times[veh] = arrival_time
            positions[veh] = customer
            unvisited.remove(customer)

        return routes_to_chromosome(routes)

    def _create_deadline_solution(self) -> Chromosome:
        """Create solution prioritizing earliest deadlines"""
        # Sort customers by deadline (spoilage time - return time)
        deadlines = []
        for i in self.feasible_customers:
            spoilage = self.nodes[i][3]
            return_time = self.distances[i][0] / self.veh_speed
            deadline = spoilage - return_time
            deadlines.append((deadline, i))

        deadlines.sort()  # Earliest deadline first

        routes = [[] for _ in range(self.veh_count)]
        loads = [0] * self.veh_count
        times = [0.0] * self.veh_count
        positions = [0] * self.veh_count

        for deadline, customer in deadlines:
            # Find best vehicle for this customer
            best_veh = None
            best_time = float('inf')

            for veh in range(self.veh_count):
                if loads[veh] >= self.veh_capa:
                    continue

                travel_time = self.distances[positions[veh]][customer] / self.veh_speed
                arrival_time = times[veh] + travel_time

                if arrival_time <= deadline and arrival_time < best_time:
                    best_time = arrival_time
                    best_veh = veh

            if best_veh is not None:
                routes[best_veh].append(customer)
                loads[best_veh] += 1
                times[best_veh] = best_time
                positions[best_veh] = customer

        return routes_to_chromosome(routes)

    def _create_random_solution(self) -> Chromosome:
        """Create random valid solution respecting capacity"""
        customers = self.feasible_customers.copy()
        random.shuffle(customers)

        routes = [[] for _ in range(self.veh_count)]
        loads = [0] * self.veh_count

        for customer in customers:
            # Find vehicles with available capacity
            available_vehs = [v for v in range(self.veh_count) if loads[v] < self.veh_capa]
            if not available_vehs:
                break  # All vehicles full

            # Assign to random available vehicle
            veh = random.choice(available_vehs)
            routes[veh].append(customer)
            loads[veh] += 1

        return routes_to_chromosome(routes)

    def _initialize_population(self) -> List[Chromosome]:
        """Initialize population with diverse solutions"""
        population = []

        # Add structured solutions
        greedy = self._create_greedy_solution()
        self._evaluate_fitness(greedy)
        population.append(greedy)

        deadline = self._create_deadline_solution()
        self._evaluate_fitness(deadline)
        population.append(deadline)

        # Fill rest with random solutions
        while len(population) < self.population_size:
            chromo = self._create_random_solution()
            self._evaluate_fitness(chromo)
            population.append(chromo)

        return population

    def _apply_mutation(self, chromo: Chromosome) -> None:
        """Apply random mutation operator"""
        mutation_type = random.randint(0, 4)

        if mutation_type == 0:
            two_opt_mutation(chromo, self.veh_count)
        elif mutation_type == 1:
            swap_mutation(chromo)
        elif mutation_type == 2:
            insert_mutation(chromo)
        elif mutation_type == 3:
            inversion_mutation(chromo)
        else:
            or_opt_mutation(chromo, self.veh_count, self.veh_capa)

    def solve(self) -> List[List[int]]:
        """
        Run genetic algorithm and return best solution

        Returns:
            List of routes, where each route is a list of customer indices
        """
        if not self.feasible_customers:
            return [[] for _ in range(self.veh_count)]

        # Initialize population
        population = self._initialize_population()

        best_fitness = float('inf')
        best_solution = None
        no_improvement_count = 0

        for gen in range(self.generations):
            # Sort by fitness
            population.sort(key=lambda c: c.fitness)

            # Track best
            if population[0].fitness < best_fitness:
                best_fitness = population[0].fitness
                best_solution = population[0].copy()
                best_solution.fitness = best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= self.early_stop_generations:
                break

            # Create new generation
            new_population = []

            # Elitism - keep best individuals
            for i in range(self.elite_count):
                elite = population[i].copy()
                elite.fitness = population[i].fitness
                new_population.append(elite)

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = tournament_selection(population, self.tournament_size)
                parent2 = tournament_selection(population, self.tournament_size)

                # Crossover (pass veh_capa to enforce capacity constraints)
                if random.random() < self.crossover_rate:
                    if random.random() < 0.7:
                        child1, child2 = order_crossover(parent1, parent2, self.veh_capa)
                    else:
                        child1, child2 = pmx_crossover(parent1, parent2, self.veh_capa)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    self._apply_mutation(child1)
                if random.random() < self.mutation_rate:
                    self._apply_mutation(child2)

                # Evaluate offspring
                self._evaluate_fitness(child1)
                self._evaluate_fitness(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        # Return best solution as routes
        if best_solution is None:
            best_solution = min(population, key=lambda c: c.fitness)

        return best_solution.decode_routes(self.veh_count)


def _solve_ga(nodes, veh_count: int, veh_capa: int, veh_speed: float,
              population_size: int = 100, generations: int = 500) -> List[List[int]]:
    """
    Solve single PVRP instance using Genetic Algorithm
    """
    ga = PVRP_GeneticAlgorithm(
        nodes=nodes,
        veh_count=veh_count,
        veh_capa=veh_capa,
        veh_speed=veh_speed,
        population_size=population_size,
        generations=generations,
    )
    return ga.solve()


def ga_solve(data, population_size: int = 100, generations: int = 500):
    """
    Solve PVRP instances using Genetic Algorithm

    Interface matches ort_solve() for comparison.

    Args:
        data: PVRP_Dataset with nodes_gen(), veh_count, veh_capa, veh_speed
        population_size: GA population size per instance
        generations: Maximum generations per instance

    Returns:
        List of solutions, where each solution is a list of routes
        routes[batch_idx][vehicle_idx] = [customer_indices...]
    """
    with Pool() as p:
        with tqdm(desc="Solving with GA", total=data.batch_size) as pbar:
            results = [
                p.apply_async(
                    _solve_ga,
                    (nodes, data.veh_count, data.veh_capa, data.veh_speed,
                     population_size, generations),
                    callback=lambda _: pbar.update()
                ) for nodes in data.nodes_gen()
            ]
            routes = [res.get() for res in results]

    return routes
