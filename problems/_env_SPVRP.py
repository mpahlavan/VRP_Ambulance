import torch
import os
from datetime import datetime


class SPVRP_Custom_Logger:
    """
    Custom format logger for Stochastic PVRP Environment
    """
    def __init__(self, log_dir=None, file_name=None):
        self.log_dir = log_dir or f"logs/spvrp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.file_name = file_name or "spvrp_log.txt"
        self.log_path = os.path.join(self.log_dir, self.file_name)

        os.makedirs(self.log_dir, exist_ok=True)

        with open(self.log_path, 'w') as f:
            f.write(f"# SPVRP Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Stochastic Perishable VRP with Travel Time Uncertainty\n\n")

    def info(self, message):
        with open(self.log_path, 'a') as f:
            f.write(f"{message}\n")


class SPVRP_Environment:
    """
    Stochastic Perishable Vehicle Routing Problem Environment

    Extends PVRP with stochastic travel times:
    - Travel speed varies randomly around nominal speed
    - Occasional "late events" cause significant slowdowns
    - Safety buffers compensate for uncertainty in pickup deadlines
    """
    VEH_STATE_SIZE = 4  # position(2), capacity(1), time(1)
    CUST_FEAT_SIZE = 4  # x, y, demand(1), spoilage_time

    def __init__(self, data, nodes=None, cust_mask=None,
                 # Reward/penalty coefficients
                 spoilage_penalty=1, unserved_penalty=1, pickup_bonus_coef=1,
                 additional_late_penalty=1, capacity_usage_coef=0.0,
                 dist_penalty_coef=0.05, idle_penalty_coef=10, success_bonus=10,
                 # Stochastic travel time parameters
                 speed_var=0.1, late_p=0.05, slow_down=0.5, late_var=0.3,
                 # Buffer parameters for uncertainty compensation
                 buffer_factor=0.2, use_conservative_feasibility=True,
                 logger=None):
        """
        Args:
            data: Problem data containing nodes, vehicle info
            nodes: Node features [batch, nodes, features]
            cust_mask: Customer availability mask

            Reward coefficients:
                spoilage_penalty: Penalty for late pickup at node
                unserved_penalty: Penalty for unserved customers
                pickup_bonus_coef: Bonus for on-time pickup
                additional_late_penalty: Extra penalty for late delivery at depot
                dist_penalty_coef: Penalty coefficient for travel distance
                idle_penalty_coef: Penalty for idle vehicles
                success_bonus: Bonus for successful completion

            Stochastic parameters:
                speed_var: Variance of speed under normal conditions (σ_normal)
                late_p: Probability of a "late event" occurring (p_late)
                slow_down: Speed multiplier during late event (μ_slow)
                late_var: Variance of speed during late event (σ_late)

            Buffer parameters:
                buffer_factor: Safety buffer as fraction of nominal travel time
                              buffer = buffer_factor * (dist / veh_speed)
                use_conservative_feasibility: Use worst-case speed for feasibility
        """
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()

        # Reward/penalty coefficients
        self.spoilage_penalty = spoilage_penalty
        self.unserved_penalty = unserved_penalty
        self.dist_penalty_coef = dist_penalty_coef
        self.pickup_bonus_coef = pickup_bonus_coef
        self.additional_late_penalty = additional_late_penalty
        self.capacity_usage_coef = capacity_usage_coef
        self.idle_penalty_coef = idle_penalty_coef
        self.success_bonus = success_bonus

        # Stochastic travel time parameters
        self.speed_var = speed_var
        self.late_p = late_p
        self.slow_down = slow_down
        self.late_var = late_var

        # Buffer parameters
        self.buffer_factor = buffer_factor
        self.use_conservative_feasibility = use_conservative_feasibility

        # Compute worst-case speed for conservative estimates
        # Worst case: late event with maximum slowdown
        self.worst_case_speed = self.veh_speed * self.slow_down * (1 - self.late_var)
        self.worst_case_speed = max(self.worst_case_speed, 0.1 * self.veh_speed)

        self.last_reward = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)

        # Initialize logger
        self.logger = logger or SPVRP_Custom_Logger()

        # Track total costs per batch
        self.batch_costs = torch.zeros(self.minibatch_size)
        self.batch_served = torch.zeros(self.minibatch_size, dtype=torch.long)

        self.late_nodes = None
        self.node_lateness_count = 0

    def _sample_speed(self):
        """
        Sample stochastic travel speed.

        Model:
        - With probability (1 - late_p): normal speed with variance speed_var
          speed = veh_speed * (1 + speed_var * N(0,1))
        - With probability late_p: slow speed due to delay event
          speed = veh_speed * slow_down * (1 + late_var * N(0,1))

        Returns:
            Sampled speed tensor [batch, 1]
        """
        late = self.nodes.new_empty((self.minibatch_size, 1)).bernoulli_(self.late_p)
        rand = torch.randn_like(late)

        speed = (late * self.slow_down * (1 + self.late_var * rand) +
                 (1 - late) * (1 + self.speed_var * rand))

        return speed.clamp_(min=0.1) * self.veh_speed

    def _compute_travel_buffer(self, dist):
        """
        Compute safety buffer for travel time uncertainty.

        The buffer compensates for potential delays in travel time.
        Effective travel time = nominal_travel_time + buffer

        Args:
            dist: Travel distance [batch, 1]

        Returns:
            buffer: Time buffer to add to nominal travel time [batch, 1]
        """
        nominal_travel_time = dist / self.veh_speed
        buffer = self.buffer_factor * nominal_travel_time
        return buffer

    def _update_vehicles(self, dest, cust_idx):
        """
        Update vehicle states after moving to destination with STOCHASTIC travel time.

        Args:
            dest: Destination node features
            cust_idx: Customer indices

        Returns:
            dist: Travel distance
            lateness: Binary indicator of lateness
            ontime_pickup: Binary indicator of on-time pickup
        """
        # Calculate travel distance
        dist = torch.pairwise_distance(
            self.cur_veh[:, 0, :2],
            dest[:, 0, :2],
            keepdim=True
        )

        # STOCHASTIC: Sample actual travel speed for this trip
        sampled_speed = self._sample_speed()
        actual_travel_time = dist / sampled_speed

        arrival_time = self.cur_veh[:, :, 3] + actual_travel_time

        # Calculate time to return to depot (also stochastic in reality,
        # but we use buffered estimate for deadline calculation)
        to_depot_dist = torch.pairwise_distance(
            dest[:, 0, :2],
            self.nodes[:, 0, :2],
            keepdim=True
        )

        # Use BUFFERED return time for deadline calculation
        # This accounts for uncertainty in the return trip
        nominal_return_time = to_depot_dist / self.veh_speed
        return_buffer = self._compute_travel_buffer(to_depot_dist)
        buffered_return_time = nominal_return_time + return_buffer

        # Calculate latest allowed arrival time (spoilage time minus buffered return time)
        # latest_arrival = spoilage_time - (return_time + buffer)
        latest_arrival = dest[:, :, 3] - buffered_return_time

        # Check if delivery was late (arrival after buffered deadline)
        lateness = (arrival_time > latest_arrival).float()
        not_late = ~(lateness.bool())
        is_customer = (cust_idx > 0).float()
        ontime_pickup = (not_late.float() * is_customer)

        # Track late nodes
        if cust_idx.any():
            self.late_nodes.scatter_(1, cust_idx, lateness.bool())
            self.node_lateness_count += lateness.sum().item()

        # Update vehicle routes for customer nodes
        for b in range(self.minibatch_size):
            node_idx = cust_idx[b, 0].item()
            if node_idx > 0:
                veh_idx = self.cur_veh_idx[b, 0].item()
                self.vehicle_routes[b, node_idx] = veh_idx

        # Capacity tracking
        old_capacity = self.cur_veh[:, :, 2].clone()

        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]  # Position x,y
        self.cur_veh[:, :, 3] += actual_travel_time  # Time (stochastic)

        # Update capacity only for customer visits
        customer_visit_mask = (cust_idx > 0).float()
        capacity_change = -1.0 * customer_visit_mask
        self.cur_veh[:, :, 2] += capacity_change

        # Update visit count
        for b in range(self.minibatch_size):
            if cust_idx[b, 0].item() > 0:
                v_idx = self.cur_veh_idx[b, 0].item()
                self.vehicle_visit_count[b, v_idx] += 1

        # Update vehicles tensor
        self.vehicles = self.vehicles.scatter(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            self.cur_veh)

        # Log vehicle paths with stochastic info
        path_info = []
        for b in range(self.minibatch_size):
            node_idx = cust_idx[b, 0].item()
            veh_idx = self.cur_veh_idx[b, 0].item()
            new_capacity = self.cur_veh[b, 0, 2].item()
            visit_count = self.vehicle_visit_count[b, veh_idx].item()

            if node_idx > 0:
                path_info.append(
                    f"PATH | B{b:03d}-V{veh_idx}: "
                    f"N{node_idx:02d} | "
                    f"Arrival={arrival_time[b, 0].item():.2f} | "
                    f"Latest={latest_arrival[b, 0].item():.2f} | "
                    f"Speed={sampled_speed[b, 0].item():.2f} | "
                    f"{'LATE' if lateness[b, 0].item() > 0 else 'On-time'} | "
                    f"Cap: {old_capacity[b, 0].item():.1f}→{new_capacity:.1f}"
                )
            else:
                path_info.append(
                    f"PATH | B{b:03d}-V{veh_idx}: "
                    f"Depot | "
                    f"Arrival={arrival_time[b, 0].item():.2f} | "
                    f"Speed={sampled_speed[b, 0].item():.2f}"
                )

        if path_info:
            self.logger.info('\n'.join(path_info))

        return dist, lateness, ontime_pickup

    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())

    def _update_mask(self, cust_idx):
        # Mark served customers
        self.served.scatter_(1, cust_idx, cust_idx > 0)

        # Capacity mask
        remaining_capacity = self.cur_veh[:, :, 2]
        capacity_mask = (remaining_capacity <= 0)
        capacity_mask = capacity_mask.unsqueeze(-1).expand(-1, -1, self.nodes_count)

        full_capacity_mask = torch.zeros_like(self.mask, dtype=torch.bool)
        for b in range(self.minibatch_size):
            v_idx = self.cur_veh_idx[b, 0].item()
            full_capacity_mask[b, v_idx] = capacity_mask[b, 0]

        # Combine masks
        self.mask = (
            self.mask |
            self.served.unsqueeze(1).expand(-1, self.veh_count, -1) |
            full_capacity_mask |
            self.veh_done.unsqueeze(-1).expand(-1, -1, self.nodes_count) |
            self.infeasible_nodes.unsqueeze(1)
        )
        self.mask[:, :, 0] = 0  # Depot always available

        # Update current vehicle mask
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))

        # Log visible nodes
        for b in range(self.minibatch_size):
            veh_idx = self.cur_veh_idx[b, 0].item()
            visible_nodes = (~self.cur_veh_mask[b, 0]).nonzero().flatten().tolist()
            visible_str = ", ".join(str(n) for n in visible_nodes if n > 0)
            if visible_str:
                self.logger.info(f"| MASK | B{b:03d} | V{veh_idx} sees nodes: [{visible_str}]")

    def _update_cur_veh(self):
        avail = self.vehicles[:, :, 3].clone()
        avail[self.veh_done] = float('inf')
        self.cur_veh_idx = avail.argmin(1, keepdim=True)
        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))

    def reset(self):
        """
        Reset environment with STOCHASTIC feasibility calculation.

        Feasibility is computed using buffered/conservative travel times
        to account for uncertainty.
        """
        # Initialize vehicles [batch, veh_count, VEH_STATE_SIZE]
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]  # Position x,y
        self.vehicles[:, :, 2] = self.veh_capa            # Capacity
        self.vehicles[:, :, 3] = 0                        # Time

        # Calculate initial feasibility mask with STOCHASTIC consideration
        depot_pos = self.nodes[:, 0:1, :2]  # [batch, 1, 2]
        node_pos = self.nodes[:, :, :2]     # [batch, nodes, 2]

        # Calculate distance from depot to each node
        to_node_dist = (node_pos - depot_pos).norm(dim=-1)  # [batch, nodes]

        if self.use_conservative_feasibility:
            # Use WORST-CASE speed for feasibility (conservative)
            # This ensures we don't plan routes that might become infeasible
            effective_speed = self.worst_case_speed
            self.logger.info(f"INIT | Using conservative feasibility with speed={effective_speed:.2f}")
        else:
            # Use buffered nominal speed
            effective_speed = self.veh_speed / (1 + self.buffer_factor)
            self.logger.info(f"INIT | Using buffered feasibility with effective_speed={effective_speed:.2f}")

        # Round trip time with conservative/buffered estimate
        round_trip_time = 2 * (to_node_dist / effective_speed)

        # Mask nodes where round trip exceeds spoilage time
        self.infeasible_nodes = round_trip_time > self.nodes[:, :, 3]  # [batch, nodes]

        # Log infeasible nodes count
        infeasible_count = self.infeasible_nodes.sum(dim=1)
        for b in range(self.minibatch_size):
            self.logger.info(f"INIT | B{b:03d} | Infeasible nodes: {infeasible_count[b].item()}")

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
        self.done = False
        self.cust_mask = self.init_cust_mask
        self.new_customers = True
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)

        # Initialize late nodes tracking
        self.late_nodes = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
        self.node_lateness_count = 0

        # Initialize vehicle visit tracking
        self.vehicle_visit_count = torch.zeros((self.minibatch_size, self.veh_count),
                                               dtype=torch.long, device=self.nodes.device)

        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
        if self.cust_mask is not None:
            self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)

        self.next_vehicle_idx = torch.zeros((self.minibatch_size,), dtype=torch.long, device=self.nodes.device)

        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype=torch.int64)
        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))

        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))

        self.vehicle_routes = self.nodes.new_zeros(
            (self.minibatch_size, self.nodes_count),
            dtype=torch.long
        ) - 1

        # Reset batch tracking
        self.batch_costs = torch.zeros(self.minibatch_size)
        self.batch_served = torch.zeros(self.minibatch_size, dtype=torch.long)

    def step(self, cust_idx):
        """Execute one step with stochastic travel time."""
        # Capacity check
        for b in range(self.minibatch_size):
            node = cust_idx[b, 0].item()
            if node > 0 and self.cur_veh[b, 0, 2] < 1:
                self.logger.info(f"CAPACITY ERROR | B{b:03d} | node {node} without capacity")
                cust_idx[b, 0] = 0

        # Update state with stochastic travel
        dest = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
        dist, node_late, node_on_time = self._update_vehicles(dest, cust_idx)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()

        # Immediate reward
        reward = (-self.spoilage_penalty * node_late
                  - self.dist_penalty_coef * dist)

        # Episode end
        if self.done:
            feasible_mask = ~self.infeasible_nodes
            feasible_mask[:, 0] = False  # Exclude depot

            # Late at pickup
            pickup_late_cnt = (self.late_nodes & feasible_mask).float().sum(dim=1, keepdim=True)

            # Late at depot (stochastic return time)
            late_at_depot_cnt = torch.zeros_like(pickup_late_cnt)
            late_at_depot_bool = torch.zeros_like(self.late_nodes, dtype=torch.bool)

            arrival_in_depot = self.vehicles[:, :, 3]

            for b in range(self.minibatch_size):
                for v in range(self.veh_count):
                    nodes_v = torch.nonzero(
                        (self.vehicle_routes[b] == v) &
                        (torch.arange(self.nodes_count, device=self.nodes.device) > 0),
                        as_tuple=False
                    ).squeeze(-1)

                    if nodes_v.numel() == 0:
                        continue

                    t_dep = arrival_in_depot[b, v].item()
                    spoil = self.nodes[b, nodes_v, 3]

                    late_flags = (t_dep > spoil)
                    late_at_depot_bool[b, nodes_v] |= late_flags
                    late_at_depot_cnt[b, 0] += late_flags.sum().float()

            # Unserved
            mask_nodes = torch.arange(self.nodes_count, device=self.nodes.device) > 0
            feasible_non_depot = feasible_mask & mask_nodes
            unserved_cnt = (~self.served & feasible_non_depot).float().sum(dim=1, keepdim=True)

            # Idle vehicles
            idle_cnt = (self.vehicle_visit_count == 0).sum(dim=1, keepdim=True)
            idle_penalty = -self.idle_penalty_coef * idle_cnt

            J_depot = -self.additional_late_penalty * late_at_depot_cnt
            J_unserv = -self.unserved_penalty * unserved_cnt
            J_idle = idle_penalty

            final_reward = J_depot + J_unserv + J_idle
            reward += final_reward

            if self.logger:
                for b in range(self.minibatch_size):
                    total_feasible = feasible_non_depot[b].sum().item()
                    total_served = self.served[b, 1:].sum().item()
                    serving_rate = (total_served / total_feasible * 100) if total_feasible > 0 else 0

                    self.logger.info(
                        f"[EPISODE_END] B{b:03d} | "
                        f"Served={total_served}/{total_feasible} ({serving_rate:.1f}%) | "
                        f"PickupLate={pickup_late_cnt[b, 0]:.0f} | "
                        f"DepotLate={late_at_depot_cnt[b, 0]:.0f} | "
                        f"Unserved={unserved_cnt[b, 0]:.0f} | "
                        f"Idle={idle_cnt[b, 0]:.0f} | "
                        f"FinalReward={reward[b, 0]:.1f}"
                    )

            self.last_reward = reward.clone()

        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        return reward

    def get_state(self):
        """Return current state"""
        return None

    def state_dict(self, dest_dict=None):
        """Save state to dictionary"""
        if dest_dict is None:
            dest_dict = {
                "vehicles": self.vehicles,
                "veh_done": self.veh_done,
                "served": self.served,
                "mask": self.mask,
                "cur_veh_idx": self.cur_veh_idx,
                "vehicle_routes": self.vehicle_routes,
                "late_nodes": self.late_nodes
            }
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["veh_done"].copy_(self.veh_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
            dest_dict["vehicle_routes"].copy_(self.vehicle_routes)
            dest_dict["late_nodes"].copy_(self.late_nodes)
        return dest_dict

    def load_state_dict(self, state_dict):
        """Load state from dictionary"""
        self.vehicles.copy_(state_dict["vehicles"])
        self.veh_done.copy_(state_dict["veh_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])
        self.vehicle_routes.copy_(state_dict["vehicle_routes"])

        if "late_nodes" in state_dict:
            self.late_nodes.copy_(state_dict["late_nodes"])

        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
