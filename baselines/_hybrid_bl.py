from marpdan.baselines._base import Baseline
import torch
import torch.nn.functional as F

class SurvivalAwareBaseline(Baseline):
    """
    
    Args:
        learner: AttentionLearner instance
        use_cumul_reward
        alpha: distance (alpha) و urgency (1-alpha)
               alpha=1.0 → pure distance-based (like nearest neighbor)
               alpha=0.0 → pure urgency-based
               alpha=0.5 → balanced
        mode: 'static' or 'dynamic'
              
    """
    
    _BIG_FLOAT = 1e9
    
    def __init__(self, learner, use_cumul_reward=False, alpha=0.5, mode='static'):
        super().__init__(learner, use_cumul_reward)
        
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha باید بین 0 و 1 باشد، دریافت شد: {alpha}")
        
        self.alpha = alpha
        self.mode = mode
        self.buf = None
        
    def _compute_distance_component(self, dyna):
        """
        محاسبه distance-based score (مانند Nearest Neighbor)
        
        Returns:
            distance_scores: [batch, 1, nodes] - کوچکتر = بهتر
        """
        # موقعیت vehicle فعلی
        veh_pos = dyna.cur_veh[:, :, :2].unsqueeze(2)  # [batch, 1, 1, 2]
        veh_pos = veh_pos.expand(-1, -1, dyna.nodes_count, -1)  # [batch, 1, nodes, 2]
        
        # موقعیت همه بیماران
        cust_pos = dyna.nodes[:, :, :2].unsqueeze(1)  # [batch, 1, nodes, 2]
        
        # فاصله اقلیدسی مربعی
        sqd = (veh_pos - cust_pos).pow(2).sum(dim=3)  # [batch, 1, nodes]
        
        # Normalize به [0, 1]
        sqd_norm = sqd / (sqd.max(dim=2, keepdim=True)[0] + 1e-8)
        
        return sqd_norm
    
    def _compute_urgency_component(self, dyna):
        """
        محاسبه survival urgency score
        
        Returns:
            urgency_scores: [batch, 1, nodes] - بزرگتر = urgent‌تر
        """
        batch_size = dyna.nodes.shape[0]
        nodes_count = dyna.nodes.shape[1]
        
        # زمان فعلی vehicle
        current_time = dyna.cur_veh[:, :, 3]  # [batch, 1]
        
        # Survival time هر بیمار
        spoilage_times = dyna.nodes[:, :, 3]  # [batch, nodes]
        
        # محاسبه pickup deadline
        # pickup_deadline = spoilage_time - time_to_hospital
        veh_pos = dyna.cur_veh[:, 0, :2]  # [batch, 2]
        depot_pos = dyna.nodes[:, 0, :2]  # [batch, 2]
        patient_pos = dyna.nodes[:, :, :2]  # [batch, nodes, 2]
        
        # فاصله از هر بیمار تا بیمارستان
        dist_to_hospital = torch.norm(
            patient_pos - depot_pos.unsqueeze(1), 
            dim=2
        )  # [batch, nodes]
        
        time_to_hospital = dist_to_hospital / dyna.veh_speed
        
        # Pickup deadline = latest time to pick up
        pickup_deadlines = spoilage_times - time_to_hospital  # [batch, nodes]
        
        # زمان باقی‌مانده تا deadline
        time_remaining = pickup_deadlines - current_time  # [batch, nodes]
        
        # Urgency: هرچه زمان کمتر، urgency بیشتر
        # استفاده از inverse با clipping برای stability
        urgency = 1.0 / (time_remaining.clamp(min=1.0) + 1e-6)  # [batch, nodes]
        
        # Normalize به [0, 1]
        urgency_norm = urgency / (urgency.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return urgency_norm.unsqueeze(1)  # [batch, 1, nodes]
    
    def _compute_dynamic_alpha(self, dyna, urgency_scores):
        """
        محاسبه α بر اساس وضعیت فعلی
        
        Strategy:
        - اگر بیماران زیادی در وضعیت critical هستند → α کوچک (focus on urgency)
        - اگر همه زمان کافی دارند → α بزرگ (focus on efficiency)
        """
        # میانگین urgency همه بیماران feasible
        mask = ~dyna.cur_veh_mask[:, 0, :]  # [batch, nodes] - True = feasible
        
        # محاسبه average urgency از بیماران feasible
        urgency_avg = (urgency_scores[:, 0, :] * mask.float()).sum(dim=1) / \
                      (mask.sum(dim=1).float() + 1e-8)  # [batch]
        
        # اگر urgency بالا → α کوچک (prioritize urgency)
        # اگر urgency پایین → α بزرگ (prioritize distance)
        alpha_dynamic = 1.0 - urgency_avg  # [batch]
        
        # Clamp به [0.2, 0.8] برای جلوگیری از extreme values
        alpha_dynamic = alpha_dynamic.clamp(0.2, 0.8)
        
        return alpha_dynamic.view(-1, 1, 1)  # [batch, 1, 1]
    
    def eval(self, dyna):
        """برای use_cumul_reward=True"""
        dyna.reset()
        return self.eval_step(dyna, None, None)
    
    def eval_step(self, dyna, learner_compat, learner_cust_idx):
        """
        محاسبه baseline value برای state فعلی
        """
        # ذخیره state
        self.buf = dyna.state_dict(self.buf)
        
        # 1. محاسبه distance component
        distance_scores = self._compute_distance_component(dyna)  # [batch, 1, nodes]
        
        # 2. محاسبه urgency component
        urgency_scores = self._compute_urgency_component(dyna)  # [batch, 1, nodes]
        
        # 3. تعیین α
        if self.mode == 'static':
            alpha = self.alpha
        elif self.mode == 'dynamic':
            alpha = self._compute_dynamic_alpha(dyna, urgency_scores)
        else:
            raise ValueError(f"mode نامعتبر: {self.mode}")
        
        # 4. ترکیب scores
        # distance: کوچکتر بهتر، urgency: بزرگتر بهتر
        # پس urgency را معکوس می‌کنیم
        combined_score = (
            alpha * distance_scores + 
            (1 - alpha) * (1.0 - urgency_scores)
        )  # [batch, 1, nodes]
        
        # 5. Add mask penalty
        combined_score = combined_score + dyna.cur_veh_mask.float() * self._BIG_FLOAT
        
        # Discourage depot unless necessary
        combined_score[:, 0, 0] += 0.5 * self._BIG_FLOAT
        
        # 6. انتخاب بهترین action و rollout
        rewards = []
        while not dyna.done:
            # Recompute scores در هر step
            distance_scores = self._compute_distance_component(dyna)
            urgency_scores = self._compute_urgency_component(dyna)
            
            if self.mode == 'dynamic':
                alpha = self._compute_dynamic_alpha(dyna, urgency_scores)
            
            combined_score = (
                alpha * distance_scores + 
                (1 - alpha) * (1.0 - urgency_scores)
            )
            
            combined_score = combined_score + dyna.cur_veh_mask.float() * self._BIG_FLOAT
            combined_score[:, 0, 0] += 0.5 * self._BIG_FLOAT
            
            # Select best action
            cust_idx = combined_score.argmin(dim=2)  # [batch, 1]
            
            # Execute action
            reward = dyna.step(cust_idx)
            rewards.append(reward)
        
        # بازگردانی state
        dyna.load_state_dict(self.buf)
        
        # Return cumulative reward as baseline value
        return torch.stack(rewards).sum(dim=0)