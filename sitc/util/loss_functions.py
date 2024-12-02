import torch
import torch.nn as nn

class AELoss(nn.Module):
    def __init__(self, alpha_rec=2, alpha_smooth=3):
        super(AELoss, self).__init__()
        self.alpha_rec = alpha_rec
        self.alpha_smooth = alpha_smooth
    
    def forward(self, inputs, targets):
        loss_rec = nn.MSELoss()
        loss_smooth = SmoothLoss()

        return self.alpha_rec * loss_rec(inputs, targets) + self.alpha_smooth * loss_smooth(inputs, targets)
    

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
    
    def forward(self, inputs, targets):
        # Permute from (B, 3, T, J) back to (B, J, T, 3)
        inputs = inputs.permute(0, 3, 2, 1)
        targets = targets.permute(0, 3, 2, 1)

        J = inputs.shape[1]
        T = inputs.shape[2]
        
        total_shifts = []
        for i in range(J-1): 
            input_shifts = []
            target_shifts = []
            for j in range(T-1):
                input_shifts.append((inputs[:, i, j] - inputs[:, i, j+1]**2))
                target_shifts.append((targets[:, i, j] - targets[:, i, j+1]**2))
            
            total_shifts.append(torch.abs(sum(input_shifts) - sum(target_shifts)))

        loss = torch.sqrt(sum(total_shifts)) / (J * T)

        return loss.mean()