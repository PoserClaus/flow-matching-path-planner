import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import WIDTH, HEIGHT


# ==========================================
# 1. DATASET LOADER (Start, Goal -> Trajectory)
# ==========================================
class PureTrajectoryDataset(Dataset):

  def __init__(self, dataset_path="expert_dataset.pt", fixed_length=40):
    if not os.path.exists(dataset_path):
      raise FileNotFoundError(
          f"'{dataset_path}' not found! Run data_generator.py first."
      )

    raw_data = torch.load(dataset_path)
    self.samples = []
    self.fixed_length = fixed_length

    for item in raw_data:
      cond = item["condition"]
      traj = item["trajectory"]  # List of (x, y) tuples

      start = cond["start"]
      goal = cond["goal"]

      # Condition vector c: [start_x, start_y, goal_x, goal_y] normalized to [0, 1]
      c_vector = [
          start[0] / WIDTH,
          start[1] / HEIGHT,
          goal[0] / WIDTH,
          goal[1] / HEIGHT,
      ]
      c_tensor = torch.tensor(c_vector, dtype=torch.float32)

      # Trajectory tensor x1: [N, 2] normalized
      traj_tensor = torch.tensor(traj, dtype=torch.float32)
      traj_tensor[:, 0] /= WIDTH
      traj_tensor[:, 1] /= HEIGHT

      # Resample trajectory to fixed_length using linear interpolation
      if len(traj_tensor) != self.fixed_length:
        indices = torch.linspace(0, len(traj_tensor) - 1, self.fixed_length)
        floor_idx = indices.long()
        ceil_idx = torch.clamp(floor_idx + 1, max=len(traj_tensor) - 1)
        frac = (indices - floor_idx.float()).unsqueeze(1)
        traj_tensor = (
            traj_tensor[floor_idx] * (1.0 - frac)
            + traj_tensor[ceil_idx] * frac
        )

      x1 = traj_tensor.flatten()  # Shape: [80] (40 points * 2 coords)
      self.samples.append((c_tensor, x1))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


# ==========================================
# 2. THE FLOW MATCHING NEURAL NETWORK
# ==========================================
class FlowMatchingMLP(nn.Module):

  def __init__(self, condition_dim=4, trajectory_dim=80, hidden_dim=256):
    super().__init__()
    # Input: x_t (trajectory) + t (time) + c (condition vector)
    input_dim = trajectory_dim + 1 + condition_dim

    self.net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, trajectory_dim),  # Predicts velocity vector v_t
    )

  def forward(self, x_t, t, c):
    net_input = torch.cat([x_t, t, c], dim=-1)
    return self.net(net_input)


# ==========================================
# 3. TRAINING LOOP WITH STRAIGHT-LINE PRIOR
# ==========================================
def train_model():
  print("Loading dataset...")
  dataset = PureTrajectoryDataset(
      dataset_path="expert_dataset.pt", fixed_length=40
  )
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

  # Condition dim = 4 (start x/y, goal x/y), Trajectory dim = 80 (40 * 2)
  model = FlowMatchingMLP(condition_dim=4, trajectory_dim=80, hidden_dim=256)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.MSELoss()

  epochs = 150
  print(f"Starting Flow Matching training for {epochs} epochs...")

  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    for c, x1 in dataloader:
      batch_size = x1.shape[0]

      # 1. Create a proper [WIDTH, HEIGHT] scaling tensor repeated for all 40 points (80 values total)
      scale_vector = torch.tensor([WIDTH, HEIGHT], device=c.device).repeat(40)

      start_xy = c[:, 0:2] * torch.tensor([WIDTH, HEIGHT], device=c.device)
      goal_xy = c[:, 2:4] * torch.tensor([WIDTH, HEIGHT], device=c.device)

      x0_list = []
      for b in range(batch_size):
        line_pts = torch.linspace(0, 1, 40, device=c.device).unsqueeze(1)
        path_line = (1.0 - line_pts) * start_xy[b] + line_pts * goal_xy[b]
        x0_list.append(path_line.flatten())

      # 2. Normalize x0 correctly using the X and Y scales
      x0 = torch.stack(x0_list) / scale_vector
      x0 = x0 + 0.05 * torch.randn_like(x1)  # Add tiny noise injection

      # 2. Sample random time t ~ U(0, 1)
      t = torch.rand(batch_size, 1, dtype=torch.float32)

      # 3. Linear interpolation path x_t
      x_t = (1.0 - t) * x0 + t * x1

      # 4. Target velocity field v_t = x1 - x0
      v_target = x1 - x0

      # 5. Predict velocity
      v_pred = model(x_t, t, c)

      # 6. Loss & Optimization
      loss = criterion(v_pred, v_target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * batch_size

    avg_loss = total_loss / len(dataset)
    if (epoch + 1) % 15 == 0 or epoch == 0:
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

  torch.save(model.state_dict(), "flow_matching_model.pt")
  print("\nTraining complete! Model saved to 'flow_matching_model.pt'.")


if __name__ == "__main__":
  train_model()