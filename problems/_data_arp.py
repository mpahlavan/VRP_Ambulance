import torch
from marpdan.problems import VRP_Dataset

class ARP_Dataset(VRP_Dataset):
    # Customer features: [x, y, demand=1, survival_time, time_window_upper_bound]
    CUST_FEAT_SIZE = 5

    @classmethod
    def generate(cls,
                 batch_size=1,
                 patient_count=100,
                 ambulance_count=25,
                 ambulance_capacity=2,
                 min_patient_count=None,
                 patient_loc_range=(0, 101),
                 survival_time_range=(30, 240),
                 speed=1):
        """
        Generate a dataset for the Ambulance Routing Problem (ARP).

        Args:
            batch_size (int): Number of instances in the batch.
            patient_count (int): Total number of patients (excluding the hospital).
            ambulance_count (int): Number of ambulances available.
            ambulance_capacity (int): Capacity of each ambulance.
            min_patient_count (int, optional): Minimum number of patients to be included.
            patient_loc_range (tuple): Range of patient locations (min, max).
            survival_time_range (tuple): Range of survival times (min, max).
            speed (float): Speed of ambulances.

        Returns:
            ARP_Dataset: An instance of the dataset class.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate random locations for patients and hospital
        # Shape: [batch_size, patient_count + 1, 2]
        locs = torch.randint(
            *patient_loc_range,
            (batch_size, patient_count + 1, 2),
            dtype=torch.float,
            device=device
        )

        # All patients have a demand of 1
        # Shape: [batch_size, patient_count, 1]
        demands = torch.ones((batch_size, patient_count, 1), dtype=torch.float, device=device)

        # Generate survival times for patients
        # Shape: [batch_size, patient_count, 1]
        survival_times = torch.randint(
            *survival_time_range,
            (batch_size, patient_count, 1),
            dtype=torch.float,
            device=device
        )

        # Extract hospital location (index 0)
        # Shape: [batch_size, 1, 2]
        hospital_loc = locs[:, 0:1, :]

        # Patient locations start from index 1
        # Shape: [batch_size, patient_count, 2]
        patient_locs = locs[:, 1:, :]

        # Calculate travel times from patients to hospital
        # Shape: [batch_size, patient_count]
        travel_times_to_hospital = torch.norm(
            patient_locs - hospital_loc.expand(-1, patient_count, -1),
            dim=2
        ) / speed  # Divided by speed to get time

        # Ensure travel_times_to_hospital has shape [batch_size, patient_count, 1]
        travel_times_to_hospital = travel_times_to_hospital.unsqueeze(2)

        # Calculate time window upper bounds (when the patient must be picked up)
        # Shape: [batch_size, patient_count, 1]
        time_window_ub = survival_times - travel_times_to_hospital

        # Combine patient features into a tensor
        # Shape: [batch_size, patient_count, CUST_FEAT_SIZE]
        patients = torch.cat((
            patient_locs,          # [batch_size, patient_count, 2]
            demands,               # [batch_size, patient_count, 1]
            survival_times,        # [batch_size, patient_count, 1]
            time_window_ub         # [batch_size, patient_count, 1]
        ), dim=2)

        # Create hospital node (depot)
        # Shape: [batch_size, 1, CUST_FEAT_SIZE]
        hospital_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE), device=device)
        hospital_node[:, :, :2] = hospital_loc  # Set hospital location
        hospital_node[:, :, 2] = 0              # Demand at hospital is zero
        hospital_node[:, :, 3] = float('inf')   # Survival time at hospital is infinity
        hospital_node[:, :, 4] = float('inf')   # Time window upper bound at hospital is infinity

        # Combine hospital and patient nodes
        # Shape: [batch_size, patient_count + 1, CUST_FEAT_SIZE]
        nodes = torch.cat((hospital_node, patients), dim=1)

        # Apply patient mask if necessary
        if min_patient_count is not None:
            # Randomly determine the actual number of patients per batch
            counts = torch.randint(
                min_patient_count,
                patient_count + 1,
                (batch_size, 1),
                dtype=torch.int64,
                device=device
            )
            # Create a mask for patients to include (True for valid patients)
            # Shape: [batch_size, patient_count]
            valid_patients_mask = torch.arange(patient_count, device=device).unsqueeze(0) < counts
            # Expand mask to match nodes shape (include hospital node)
            # Shape: [batch_size, patient_count + 1]
            patient_mask = torch.cat((
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),  # Include hospital
                valid_patients_mask
            ), dim=1)
            # Zero out nodes that are masked (invalid patients)
            nodes = nodes * patient_mask.unsqueeze(2).float()
        else:
            patient_mask = None

        # Create dataset instance
        dataset = cls(
            veh_count=ambulance_count,
            veh_capa=ambulance_capacity,
            veh_speed=speed,
            nodes=nodes,
            cust_mask=patient_mask
        )
        return dataset
