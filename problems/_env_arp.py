import torch
from marpdan.problems import VRP_Environment

class ARP_Environment(VRP_Environment):
    # Vehicle state: [x, y, remaining_capacity, current_time, patients_onboard]
    # We will extend the state to include survival times of onboard patients
    # For simplicity, we'll assume a maximum capacity (max_patients_onboard) for tensor sizes
    VEH_STATE_SIZE = 5  # [x, y, capacity, current_time, patients_onboard]
    CUST_FEAT_SIZE = 5  # [x, y, demand=1, survival_time, time_window_upper_bound]

    def __init__(self, data, nodes=None, patient_mask=None,
                 gamma=1.0,     # Coefficient for late arrival penalty
                 sigma=1.0,     # Coefficient for early arrival penalty
                 pending_cost=1000,
                 vacancy_coefficient=100):  # Penalty coefficient for empty capacity
        super().__init__(data, nodes, patient_mask, pending_cost)
        self.gamma = gamma
        self.sigma = sigma
        self.vacancy_coefficient = vacancy_coefficient

        # Maximum capacity for onboard patients (assumed known)
        self.max_patients_onboard = self.veh_capa

        # Initialize data structures to store survival times of onboard patients
        self.onboard_survival_times = None  # Will be initialized in reset()

    def _sample_speed(self):
        # Speed is constant in this model
        return self.veh_speed

    def reset(self):
        """Initialize environment state."""
        super().reset()

        # Extend vehicle state to include survival times of onboard patients
        # Shape: [batch_size, veh_count, max_patients_onboard]
        self.onboard_survival_times = self.nodes.new_full(
            (self.minibatch_size, self.veh_count, self.max_patients_onboard),
            fill_value=float('inf')
        )

        # Initialize vehicles at the hospital
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2].expand(-1, self.veh_count, -1)  # Position at hospital
        self.vehicles[:, :, 2] = self.veh_capa  # Full capacity
        self.vehicles[:, :, 3] = 0.0            # Current time
        self.vehicles[:, :, 4] = 0              # Patients onboard

        # Initialize other state variables
        self.veh_done = torch.zeros((self.minibatch_size, self.veh_count), dtype=torch.bool, device=self.nodes.device)
        self.done = False
        self.served = torch.zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool, device=self.nodes.device)
        self.mask = torch.zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool, device=self.nodes.device)

        if self.init_cust_mask is not None:
            self.mask |= self.init_cust_mask.unsqueeze(1).expand(-1, self.veh_count, -1)

        # Update current vehicle indices and masks
        self._update_cur_veh()
        self._update_mask()

    def _update_cur_veh(self):
        """Update current vehicle based on availability."""
        # Select the next available vehicle (e.g., the one with the earliest current time)
        avail = self.vehicles[:, :, 3].clone()  # Current times
        avail[self.veh_done] = float('inf')     # Mark done vehicles as unavailable
        self.cur_veh_idx = avail.argmin(dim=1).unsqueeze(1)  # Shape: [batch_size, 1]
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx.unsqueeze(2).expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx.unsqueeze(2).expand(-1, -1, self.nodes_count))

    def _update_mask(self):
        """Update mask based on time windows, survival times, and feasibility."""
        current_time = self.cur_veh[:, :, 3]  # Shape: [batch_size, 1]
        current_positions = self.cur_veh[:, :, :2]  # Shape: [batch_size, 1, 2]

        # Compute survival times and time windows
        survival_times = self.nodes[:, :, 3]  # Shape: [batch_size, nodes_count]
        time_window_ub = self.nodes[:, :, 4]  # Shape: [batch_size, nodes_count]

        # Calculate travel times to all nodes
        node_positions = self.nodes[:, :, :2]  # Shape: [batch_size, nodes_count, 2]
        travel_times = torch.norm(
            current_positions.unsqueeze(2) - node_positions.unsqueeze(1),
            dim=-1
        ) / self._sample_speed()  # Shape: [batch_size, 1, nodes_count]

        # Calculate arrival times at nodes
        arrival_times = current_time.unsqueeze(2) + travel_times  # Shape: [batch_size, 1, nodes_count]

        # Mask nodes where survival times have expired
        survival_mask = arrival_times > survival_times.unsqueeze(1)  # Shape: [batch_size, 1, nodes_count]

        # Mask nodes where time window upper bounds have expired
        time_window_mask = arrival_times > time_window_ub.unsqueeze(1)  # Shape: [batch_size, 1, nodes_count]

        # Mask nodes that have already been served or where vehicle is done
        served_mask = self.served.unsqueeze(1)  # Shape: [batch_size, 1, nodes_count]
        veh_done_mask = self.veh_done.unsqueeze(2)  # Shape: [batch_size, veh_count, 1]

        # Mask for capacity overload (cannot pick up if capacity is insufficient)
        demands = self.nodes[:, :, 2]  # Shape: [batch_size, nodes_count]
        capacity_mask = self.cur_veh[:, :, 2].unsqueeze(2) < demands.unsqueeze(1)  # Shape: [batch_size, 1, nodes_count]

        # Feasibility mask: Check if adding a node violates onboard patients' survival times
        # Calculate projected arrival time at hospital after visiting each node
        hospital_pos = self.nodes[:, 0:1, :2]  # Shape: [batch_size, 1, 2]
        travel_to_hospital = torch.norm(
            node_positions.unsqueeze(1) - hospital_pos.unsqueeze(2),
            dim=-1
        ) / self._sample_speed()  # Shape: [batch_size, 1, nodes_count]
        total_times_to_hospital = arrival_times + 0 + travel_to_hospital  # Assuming zero service time

        # Get minimum survival times among onboard patients
        min_survival_times, _ = self.onboard_survival_times.min(dim=2)  # Shape: [batch_size, veh_count]
        min_survival_times = min_survival_times.gather(1, self.cur_veh_idx)  # Shape: [batch_size, 1]
        min_survival_times[min_survival_times == float('inf')] = float('inf')  # Handle no patients onboard

        # Feasibility mask: Violates survival time of onboard patients
        feasibility_mask = total_times_to_hospital > min_survival_times.unsqueeze(2)  # Shape: [batch_size, 1, nodes_count]

        # Combine all masks
        combined_mask = survival_mask | time_window_mask | served_mask | veh_done_mask | capacity_mask | feasibility_mask

        # Update mask (exclude depot from being masked)
        self.mask.scatter_(1, self.cur_veh_idx.unsqueeze(2).expand(-1, -1, self.nodes_count), combined_mask)
        self.mask[:, :, 0] = False  # Depot is always available

    def _update_vehicles(self, dest):
        """Update vehicle states after moving to a destination node."""
        # Calculate travel distance and time to destination
        dist = torch.norm(self.cur_veh[:, :, :2] - dest[:, :, :2], dim=2, keepdim=True)  # Shape: [batch_size, 1, 1]
        tt = dist / self._sample_speed()  # Shape: [batch_size, 1, 1]

        # Update current time after moving to destination
        arrival_time_at_dest = self.cur_veh[:, :, 3].unsqueeze(2) + tt  # Shape: [batch_size, 1, 1]
        service_time = 0  # Assuming zero service time at nodes

        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]  # Update position
        self.cur_veh[:, :, 3] = arrival_time_at_dest.squeeze(2) + service_time  # Update current time

        # Check if destination is the hospital
        is_hospital = (dest[:, :, 2] == 0)  # Shape: [batch_size, 1]

        # Update capacity and patients onboard
        demand = dest[:, :, 2]  # Shape: [batch_size, 1]
        self.cur_veh[:, :, 2] -= demand.squeeze(1)  # Update remaining capacity

        # For pickups, add patient survival times to onboard_survival_times
        for b in range(self.minibatch_size):
            v_idx = self.cur_veh_idx[b, 0]
            if demand[b, 0] > 0 and not is_hospital[b, 0]:
                # Find first empty slot in onboard_survival_times
                empty_slots = (self.onboard_survival_times[b, v_idx] == float('inf'))
                if empty_slots.any():
                    idx = empty_slots.nonzero(as_tuple=False)[0, 0]
                    self.onboard_survival_times[b, v_idx, idx] = dest[b, 0, 3]  # Survival time
                    self.cur_veh[b, 0, 4] += 1  # Increment patients onboard

        # Calculate penalties if arrived at hospital
        penalties = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)
        if is_hospital.any():
            arrived_at_hospital = is_hospital.squeeze(1)
            penalties[arrived_at_hospital] = self._calculate_penalties(
                self.cur_veh[arrived_at_hospital, :, 3],
                self.cur_veh_idx[arrived_at_hospital]
            )

            # Reset onboard patients and survival times for vehicles that returned to hospital
            for b_idx in arrived_at_hospital.nonzero(as_tuple=False).squeeze(1).tolist():
                v_idx = self.cur_veh_idx[b_idx, 0]
                self.onboard_survival_times[b_idx, v_idx] = float('inf')
                self.cur_veh[b_idx, 0, 4] = 0  # Reset patients onboard
                self.cur_veh[b_idx, 0, 2] = self.veh_capa  # Reset capacity

        # Update vehicles tensor
        self.vehicles.scatter_(
            1,
            self.cur_veh_idx.unsqueeze(2).expand(-1, -1, self.VEH_STATE_SIZE),
            self.cur_veh
        )

        # Calculate reward: negative distance plus penalties
        reward = -dist.squeeze(2) + penalties

        return reward

    def _calculate_penalties(self, arrival_times, veh_indices):
        """Calculate penalties based on arrival times and onboard patients' survival times."""
        total_penalty = torch.zeros_like(arrival_times)

        for b in range(arrival_times.size(0)):
            v_idx = veh_indices[b, 0]
            veh_arrival_time = arrival_times[b]

            # Get survival times of onboard patients
            survival_times = self.onboard_survival_times[b, v_idx]
            valid_survival_times = survival_times[survival_times != float('inf')]

            # Calculate penalties for each patient
            for T_surv in valid_survival_times:
                T_diff = veh_arrival_time - T_surv

                if T_diff >= 0:
                    # Patient died en route
                    TD = T_diff
                    PD = 1 + self.gamma * (TD ** 2)
                    total_penalty[b] -= PD  # Penalty is subtracted
                else:
                    # Patient arrived early
                    TR = T_diff
                    PR = self.sigma / abs(TR)
                    total_penalty[b] += PR  # Reward is added

        return total_penalty.unsqueeze(1)  # Ensure shape matches reward

    def step(self, cust_idx):
        """Perform a step by moving the current vehicle to the selected customer index."""
        dest = self.nodes.gather(
            1,
            cust_idx.unsqueeze(2).expand(-1, -1, self.CUST_FEAT_SIZE)
        )  # Shape: [batch_size, 1, CUST_FEAT_SIZE]
        reward = self._update_vehicles(dest)

        # Mark customers as served
        served_customers = cust_idx.squeeze(1)  # Shape: [batch_size]
        self.served.scatter_(1, served_customers.unsqueeze(1), True)

        # Update vehicle done status
        self.veh_done.scatter_(1, self.cur_veh_idx, served_customers == 0)

        # Update mask and current vehicle
        self._update_cur_veh()
        self._update_mask()

        # Check if all vehicles are done
        self.done = self.veh_done.all()

        # If done, apply penalties for unserved patients and vacant capacities
        if self.done:
            total_penalty = self.calculate_total_penalty()
            reward -= total_penalty.unsqueeze(1)

        return reward

    def calculate_total_penalty(self):
        """Calculate penalties for unserved patients and vacant capacities."""
        # Penalty for unserved patients
        unserved_patients = (~self.served).float().sum(dim=1) - 1  # Exclude depot
        pending_patients_penalty = self.pending_cost * unserved_patients

        # Penalty for vacant capacities if there are unserved patients
        total_vacant_capacity = self.vehicles[:, :, 2].sum(dim=1)  # Shape: [batch_size]
        vacancy_penalty = self.vacancy_coefficient * total_vacant_capacity

        # Apply vacancy penalty only if there are unserved patients
        vacancy_penalty = torch.where(unserved_patients > 0, vacancy_penalty, torch.zeros_like(vacancy_penalty))

        # Total penalty
        total_penalty = pending_patients_penalty + vacancy_penalty

        # Sum over batch
        total_penalty = total_penalty

        return total_penalty

    def state_dict(self, dest_dict=None):
        dest_dict = super().state_dict(dest_dict)
        # Save onboard survival times
        dest_dict["onboard_survival_times"] = self.onboard_survival_times
        return dest_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.onboard_survival_times = state_dict["onboard_survival_times"]
