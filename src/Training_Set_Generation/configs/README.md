# Generating training data

Run everything from `src/Training_Set_Generation`:

```
cd src/Training_Set_Generation

python Generate_Testset.py --n-scenes 2                 # 1. quick test
python Generate_Testset.py                              # 2. generate with configs/default.json
python Plot_Testset.py Test_Sets/default.npz            # 3. browse the result (← → pair, ↑ ↓ scene)
python Plot_Testset.py Test_Sets/default.npz --pdf all.pdf   # or: all scenes as a scrollable PDF
```

Own experiment: copy `configs/default.json`, change what you want, set a new `out`, then
`python Generate_Testset.py --config configs/my_test.json`.

Options: `--config` (config file), `--out` (output file), `--n-scenes`, `--seed`,
`--overwrite` (replace an existing output file; without it the generator stops instead of overwriting).

Load in training:

```python
from Dataset_Format import load_dataset
d = load_dataset("Test_Sets/default.npz")
d["obs/lidar"], d["goal_rel"], d["traj"], d["weight"], d["scene_id"]
```

Fields are explained in `Dataset_Format.py`. Split train/val by `scene_id`, not by sample.

# Config

Units: metres, radians. `[a, b]` = random value between a and b.

| Key | Meaning |
|---|---|
| `seed` | Random seed. Same seed = same dataset. |
| `n_scenes` | Number of random scenes. |
| `out` | Output file. |
| `car.radius` | Robot size (modelled as a circle). |
| `scene.type` | Scene generator. |
| `scene.params.n_boxes` | Number of boxes per scene. |
| `scene.params.require_blocking` | Keep `false` (has no effect on the dataset). |
| `pairs.per_scene` | Start/goal pairs per scene. |
| `pairs.min_dist` | Minimum distance between start and goal. |
| `pairs.min_clearance` | Minimum free space around start and goal. |
| `solver.type` | Path solver. |
| `solver.params.resolution` | Grid cell size. Smaller = more precise, slower. |
| `solver.params.max_paths` | Max. number of different paths per pair. |
| `solver.params.min_path_distance` | How far apart two paths must be to count as different. |
| `solver.params.max_length_ratio` | Drop paths longer than this × the shortest. |
| `solver.params.length_tau` | Preference for short paths. Smaller = stronger preference. |
| `solver.params.safety_margin` | Extra distance to obstacles. |
| `sensors` | Sensors measured at the start. Output: `obs/<type>`. |
| `sensors[lidar].n_beams` | Number of LiDAR beams. |
| `sensors[lidar].max_range` | LiDAR range. |
| `trajectory.n_points` | Points per path. |

More parameters (box sizes, sensor noise, …): see the `Config` class in the matching file
in `Scenes/`, `Solvers/` or `Sensors/`.

Runtime: ~0.25 s per pair → default config ≈ 1 min. Test with `--n-scenes 2` first.
