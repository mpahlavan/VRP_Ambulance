"""
Genetic Algorithm Operators for PVRP
VRP-specific crossover and mutation operators
"""

import random
from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class Chromosome:
    """Represents a PVRP solution using giant tour encoding"""
    genes: List[int]                          # Giant tour with depot (0) as delimiter
    fitness: float = float('inf')             # Cost to minimize (lower is better)
    routes: List[List[int]] = field(default_factory=list)  # Decoded routes per vehicle

    def decode_routes(self, veh_count: int) -> List[List[int]]:
        """
        Split giant tour into vehicle routes

        Example: [3, 1, 4, 0, 2, 6, 5, 0] with veh_count=2
        Returns: [[3, 1, 4], [2, 6, 5]]
        """
        routes = []
        current_route = []

        for gene in self.genes:
            if gene == 0:  # Depot delimiter
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(gene)

        # Don't forget last route if no trailing delimiter
        if current_route:
            routes.append(current_route)

        # Pad with empty routes if fewer routes than vehicles
        while len(routes) < veh_count:
            routes.append([])

        # Truncate if more routes than vehicles (shouldn't happen normally)
        self.routes = routes[:veh_count]
        return self.routes

    def copy(self) -> 'Chromosome':
        """Create a deep copy"""
        return Chromosome(
            genes=self.genes.copy(),
            fitness=float('inf'),  # Reset fitness for re-evaluation
            routes=[]
        )


def routes_to_chromosome(routes: List[List[int]]) -> Chromosome:
    """Convert route list to giant tour chromosome"""
    genes = []
    for route in routes:
        genes.extend(route)
        genes.append(0)  # Depot delimiter
    return Chromosome(genes=genes)


def tournament_selection(population: List[Chromosome], tournament_size: int = 5) -> Chromosome:
    """
    Tournament selection - select best from random subset

    Args:
        population: List of chromosomes (should be evaluated)
        tournament_size: Number of individuals in tournament

    Returns:
        Selected chromosome (copy)
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    winner = min(tournament, key=lambda c: c.fitness)
    return winner.copy()


def order_crossover(parent1: Chromosome, parent2: Chromosome, veh_capa: int = None) -> Tuple[Chromosome, Chromosome]:
    """
    Order Crossover (OX) - preserves relative order of customers

    Good for TSP/VRP as it maintains adjacency information.
    Operates on customer indices only (removes delimiters, then reinserts).

    Args:
        parent1, parent2: Parent chromosomes
        veh_capa: Vehicle capacity (max customers per route)

    Returns:
        Two offspring chromosomes
    """
    # Extract customer genes (exclude depot delimiters)
    genes1 = [g for g in parent1.genes if g != 0]
    genes2 = [g for g in parent2.genes if g != 0]

    # Handle edge cases
    if not genes1 or not genes2:
        return parent1.copy(), parent2.copy()

    size = len(genes1)
    if size <= 2:
        return parent1.copy(), parent2.copy()

    # Ensure both parents have same customers
    if set(genes1) != set(genes2):
        # Different customer sets - return copies
        return parent1.copy(), parent2.copy()

    # Select two crossover points
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size - 1)

    # Create children
    child1_genes = [None] * size
    child2_genes = [None] * size

    # Copy segment from parents
    child1_genes[cx1:cx2+1] = genes1[cx1:cx2+1]
    child2_genes[cx1:cx2+1] = genes2[cx1:cx2+1]

    # Fill remaining positions from other parent
    def fill_remaining(child: List, donor: List, cx1: int, cx2: int):
        donor_pos = (cx2 + 1) % size
        child_pos = (cx2 + 1) % size

        while None in child:
            gene = donor[donor_pos]
            if gene not in child:
                child[child_pos] = gene
                child_pos = (child_pos + 1) % size
            donor_pos = (donor_pos + 1) % size

    fill_remaining(child1_genes, genes2, cx1, cx2)
    fill_remaining(child2_genes, genes1, cx1, cx2)

    # Count original delimiters to maintain similar route structure
    n_delimiters = parent1.genes.count(0)

    # Insert delimiters back, respecting capacity
    child1 = Chromosome(genes=_insert_delimiters(child1_genes, n_delimiters, veh_capa))
    child2 = Chromosome(genes=_insert_delimiters(child2_genes, n_delimiters, veh_capa))

    return child1, child2


def _insert_delimiters(genes: List[int], n_delimiters: int, max_per_route: int = None) -> List[int]:
    """
    Insert depot delimiters (0) to split customers into routes

    Strategy: Distribute delimiters to respect capacity constraints

    Args:
        genes: List of customer indices
        n_delimiters: Minimum number of delimiters to insert
        max_per_route: Maximum customers per route (vehicle capacity)
    """
    if not genes:
        return [0]  # Empty route

    result = []
    n_customers = len(genes)

    # Calculate customers per route, respecting capacity if provided
    if max_per_route is not None and max_per_route > 0:
        customers_per_route = max_per_route
    elif n_delimiters > 0:
        customers_per_route = max(1, n_customers // n_delimiters)
    else:
        customers_per_route = n_customers  # All in one route if no delimiters

    current_route_size = 0
    for gene in genes:
        result.append(gene)
        current_route_size += 1

        # Insert delimiter when route is full
        if current_route_size >= customers_per_route:
            result.append(0)
            current_route_size = 0

    # Ensure we have trailing delimiter
    if result[-1] != 0:
        result.append(0)

    return result


def pmx_crossover(parent1: Chromosome, parent2: Chromosome, veh_capa: int = None) -> Tuple[Chromosome, Chromosome]:
    """
    Partially Mapped Crossover (PMX)

    Alternative crossover that maintains absolute positions better.

    Args:
        parent1, parent2: Parent chromosomes
        veh_capa: Vehicle capacity (max customers per route)
    """
    genes1 = [g for g in parent1.genes if g != 0]
    genes2 = [g for g in parent2.genes if g != 0]

    if not genes1 or not genes2 or set(genes1) != set(genes2):
        return parent1.copy(), parent2.copy()

    size = len(genes1)
    if size <= 2:
        return parent1.copy(), parent2.copy()

    # Select crossover points
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size - 1)

    # Initialize children as copies
    child1 = genes1.copy()
    child2 = genes2.copy()

    # Create mapping
    for i in range(cx1, cx2 + 1):
        val1, val2 = genes1[i], genes2[i]

        # Swap in child1
        idx1 = child1.index(val1)
        idx2 = child1.index(val2)
        child1[idx1], child1[idx2] = child1[idx2], child1[idx1]

        # Swap in child2
        idx1 = child2.index(val1)
        idx2 = child2.index(val2)
        child2[idx1], child2[idx2] = child2[idx2], child2[idx1]

    n_delimiters = parent1.genes.count(0)
    return (
        Chromosome(genes=_insert_delimiters(child1, n_delimiters, veh_capa)),
        Chromosome(genes=_insert_delimiters(child2, n_delimiters, veh_capa))
    )


def two_opt_mutation(chromo: Chromosome, veh_count: int) -> None:
    """
    2-opt local search mutation

    Reverses a segment within a single route to potentially reduce distance.
    Modifies chromosome in place.
    """
    routes = chromo.decode_routes(veh_count)

    # Find routes with at least 2 customers
    valid_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 2]
    if not valid_routes:
        return

    # Select a random route
    route_idx, route = random.choice(valid_routes)

    # Select two positions to reverse between
    i = random.randint(0, len(route) - 2)
    j = random.randint(i + 1, len(route) - 1)

    # Reverse segment
    routes[route_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]

    # Rebuild giant tour
    new_genes = []
    for r in routes:
        new_genes.extend(r)
        new_genes.append(0)
    chromo.genes = new_genes
    chromo.fitness = float('inf')  # Reset for re-evaluation


def swap_mutation(chromo: Chromosome) -> None:
    """
    Swap mutation - exchange two random customers

    Can swap customers between different routes.
    Modifies chromosome in place.
    """
    # Find positions of customers (not delimiters)
    customer_positions = [i for i, g in enumerate(chromo.genes) if g != 0]

    if len(customer_positions) < 2:
        return

    # Select two random positions
    i, j = random.sample(customer_positions, 2)

    # Swap
    chromo.genes[i], chromo.genes[j] = chromo.genes[j], chromo.genes[i]
    chromo.fitness = float('inf')


def insert_mutation(chromo: Chromosome) -> None:
    """
    Insert mutation - remove a customer and insert at random position

    Modifies chromosome in place.
    """
    customer_positions = [i for i, g in enumerate(chromo.genes) if g != 0]

    if len(customer_positions) < 2:
        return

    # Select customer to move
    src_idx = random.choice(customer_positions)
    gene = chromo.genes.pop(src_idx)

    # Find new valid positions (not at very end after last delimiter)
    # Insert before a random position
    dst_idx = random.randint(0, len(chromo.genes))
    chromo.genes.insert(dst_idx, gene)
    chromo.fitness = float('inf')


def inversion_mutation(chromo: Chromosome) -> None:
    """
    Inversion mutation - reverse a random segment of the chromosome

    Can span across route delimiters.
    Modifies chromosome in place.
    """
    if len(chromo.genes) < 3:
        return

    # Select segment to invert
    i = random.randint(0, len(chromo.genes) - 2)
    j = random.randint(i + 1, len(chromo.genes) - 1)

    # Invert
    chromo.genes[i:j+1] = chromo.genes[i:j+1][::-1]
    chromo.fitness = float('inf')


def or_opt_mutation(chromo: Chromosome, veh_count: int, veh_capa: int = None) -> None:
    """
    Or-opt mutation - relocate a sequence of 1-3 consecutive customers

    Modifies chromosome in place.

    Args:
        chromo: Chromosome to mutate
        veh_count: Number of vehicles
        veh_capa: Vehicle capacity (if provided, respects capacity constraints)
    """
    routes = chromo.decode_routes(veh_count)

    # Find routes with customers
    valid_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 1]
    if not valid_routes:
        return

    # Select source route
    src_route_idx, src_route = random.choice(valid_routes)

    # Select sequence length (1-3)
    seq_len = min(random.randint(1, 3), len(src_route))

    # Select start position
    start = random.randint(0, len(src_route) - seq_len)

    # Extract sequence
    sequence = src_route[start:start + seq_len]

    # Remove from source
    new_src_route = src_route[:start] + src_route[start + seq_len:]
    routes[src_route_idx] = new_src_route

    # Select destination route (can be same or different)
    # If capacity is provided, prefer routes that have space
    if veh_capa is not None:
        # Find routes with enough capacity
        available_routes = [i for i in range(len(routes))
                          if len(routes[i]) + seq_len <= veh_capa or i == src_route_idx]
        if available_routes:
            dst_route_idx = random.choice(available_routes)
        else:
            dst_route_idx = src_route_idx  # Put back in source if no space
    else:
        dst_route_idx = random.randint(0, len(routes) - 1)

    dst_route = routes[dst_route_idx]

    # Select insertion position
    insert_pos = random.randint(0, len(dst_route))

    # Insert sequence
    routes[dst_route_idx] = dst_route[:insert_pos] + sequence + dst_route[insert_pos:]

    # Rebuild chromosome
    new_genes = []
    for r in routes:
        new_genes.extend(r)
        new_genes.append(0)
    chromo.genes = new_genes
    chromo.fitness = float('inf')
