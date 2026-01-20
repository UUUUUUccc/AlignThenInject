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

git clone https://github.com/Jiaxuan-Li/EVCap.git

# You can name the env as you like; here we use "xxx" as an example
conda env create -f EVCap/environment.yaml -n  xxx
conda activate xxx

data/coco/
  images/
    train2017/
    val2017/
  annotations/
    annotations/
      captions_train2017.json
      captions_val2017.json
      instances_train2017.json
      instances_val2017.json
      ...
4. Training (Slurm / sbatch)

This repo provides two Slurm scripts:

task_align.slurm: Stage 1 (Align)

task_inject.slurm: Stage 2 (Inject)

4.1 Edit key fields

Before submission, update:

conda env name (e.g., xxx)

DATA_ROOT (COCO root path)

MODEL_ROOT (weights/models path)

OUTDIR / CKPT_DIR (recommended checkpoints/ or your desired output path)

Slurm resources: GPUs, partition, walltime, memory, etc.

4.2 Submit jobs
# Stage 1: Align
sbatch task_align.slurm

# Stage 2: Inject (submit after Stage 1 finishes successfully)
sbatch task_inject.slurm


Checkpoints will be saved under checkpoints/ (or the directory specified in your scripts).



6. Acknowledgements

This project is built upon EVCap, and follows its environment setup and evaluation conventions.

7. Citation

If you use EVCap in your work, please cite EVCap (example):

@article{li2024evcap,
  title={EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension},
  author={Jiaxuan Li and Duc Minh Vo and Akihiro Sugimoto and Hideki Nakayama},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
}

