# AlignThenInject

This repository is **built upon EVCap** and follows its overall training/evaluation conventions, with a two-stage training pipeline: **Align → Inject**.

- **Environment**: reuse EVCap conda environment (`environment.yaml`)
- **Data**: MS COCO (images + annotations)
- **Training**: Slurm jobs via `sbatch` (`task_align.slurm` / `task_inject.slurm`)
- **Checkpoints**: saved under `checkpoints/` by default (or the output directory specified in Slurm scripts)
- **Evaluation**: follow EVCap-style evaluation scripts and interfaces

> Note: Paths and Slurm resources vary across clusters. Please edit the slurm scripts to match your setup (data paths, model paths, output paths, conda env name, etc.).

---

## 1. Setup

### 1.1 Clone

```bash
git clone https://github.com/UUUUUUccc/AlignThenInject.git
cd AlignThenInject
