import os
import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
import random
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.common.metric_logger import MetricLogger, SmoothedValue
from utils.training.optims import LinearWarmupCosineLRScheduler, set_optimizer
from utils.common.dist_utils import get_rank, init_distributed_mode, get_world_size
from models.two_stage.align_then_inject import AlignThenInject
from utils.data.dataset import COCOCaptionDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, cur_epoch, output_dir):
    model_no_ddp = model.module if hasattr(model, 'module') else model
    
    trainable_params = {}
    for name, param in model_no_ddp.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
    
    save_obj = {
        "model": {name: param for name, param in trainable_params.items()},
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
        "model_config": {
            "stage": model_no_ddp.stage,
        }
    }
    
    checkpoint_path = os.path.join(output_dir, f"stage2_epoch_{cur_epoch:03d}.pt")
    print("Saving stage2 checkpoint at epoch {} to {}.".format(cur_epoch, checkpoint_path))
    torch.save(save_obj, checkpoint_path)
    
    return checkpoint_path


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, args, device):
    model.train()
    
    if args.amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    
    header = f'Stage2 Caption Epoch: [{epoch}]'
    print_freq = 50
    
    for idx, samples in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        samples['image'] = samples['image'].to(device)
        
        scheduler.step(cur_epoch=epoch, cur_step=idx)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = model(samples)["loss"]
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Caption Training')
    
    parser.add_argument('--output_dir', default='', help='')
    parser.add_argument('--data_root', default='', help='')
    parser.add_argument('--ext_path', default='', help='')
    parser.add_argument('--stage1_checkpoint', default='', help='')
    
    parser.add_argument('--vit_model', default='eva_clip_g', help='')
    parser.add_argument('--llama_model', default='', help='')
    parser.add_argument('--q_former_model', default='', help='')
    parser.add_argument('--prompt_path', default='', help='')
    
    parser.add_argument('--topn', type=int, default=9, help='')
    parser.add_argument('--num_query_token_txt', type=int, default=8, help='')
    
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=6, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='')
    parser.add_argument('--min_lr', type=float, default=8e-5, help='')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='')
    parser.add_argument('--amp', action='store_true', default=True, help='')
    
    parser.add_argument('--world_size', type=int, default=1, help='')
    parser.add_argument('--rank', type=int, default=0, help='')
    parser.add_argument('--dist_url', default='env://', help='')
    parser.add_argument('--distributed', action='store_true', default=False, help='')
    
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--save_freq', type=int, default=1, help='')
    parser.add_argument('--resume', default='', help='')
    parser.add_argument('--log_filter_every', type=int, default=0, help='')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    if args.distributed:
        init_distributed_mode(args)
        device = torch.device(f"cuda:{get_rank()}")
        print(f"Process {get_rank()}: Using device {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("Creating AlignThenInject for Stage 2 (Caption)")
    print("="*80)
    model = AlignThenInject(
        ext_path=args.ext_path,
        vit_model=args.vit_model,
        q_former_model=args.q_former_model,
        llama_model=args.llama_model,
        prompt_path=args.prompt_path,
        topn=args.topn,
        num_query_token_txt=args.num_query_token_txt,
        stage="caption",
    )
    
    if args.stage1_checkpoint:
        success = model.load_stage1_weights(args.stage1_checkpoint)
        if success:
            print(f"\nSuccessfully loaded Stage 1 weights from: {args.stage1_checkpoint}")
        else:
            print("\nFailed to load Stage 1 weights, training from scratch...")
    else:
        print("\nNo Stage 1 checkpoint specified, training from scratch...")
    
    model.to(device)

    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.1%}\n")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Stage 2 Trainable: {name} - {param.shape}")
    
    optimizer = set_optimizer(
        model=model,
        init_lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.999,
    )
    
    print("\nCreating COCO Caption dataset...")
    train_dataset = COCOCaptionDataset(
        data_root=args.data_root,
        split='train'
    )
    
    if args.distributed:
        sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=args.epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=args.min_lr,
        init_lr=args.lr,
        warmup_start_lr=args.warmup_start_lr,
        warmup_steps=args.warmup_steps,
    )
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[get_rank()]
        )
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            model_state_dict = checkpoint.get('model', {})
            model_no_ddp = model.module if hasattr(model, 'module') else model
            
            loaded_keys = []
            for key in model_state_dict:
                if hasattr(model_no_ddp, key.split('.')[0]):
                    try:
                        model_no_ddp.load_state_dict({key: model_state_dict[key]}, strict=False)
                        loaded_keys.append(key)
                    except:
                        pass
            
            print(f"Loaded {len(loaded_keys)} parameters from checkpoint")
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    print(f"\nStarting Stage 2 Caption training for {args.epochs} epochs")
    print("Progressive Knowledge-Augmented Training - Stage 2: Caption Generation\n")
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            device=device
        )
        
        if get_rank() == 0 and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, args.output_dir)
        
        if get_rank() == 0:
            print(f"Epoch {epoch} completed:")
            print(f"  Loss: {train_stats['loss']:.4f}")
            print(f"  LR: {train_stats['lr']:.6f}")
    
    if get_rank() == 0:
        final_checkpoint = save_checkpoint(model, optimizer, args.epochs - 1, args.output_dir)


if __name__ == '__main__':
    main()
