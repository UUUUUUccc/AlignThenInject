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
from models.two_stage.AlignThenInject import AlignThenInject
from utils.data.dataset import VQADataset


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
    
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
    }
    
    checkpoint_path = os.path.join(output_dir, f"stage1_vqa_epoch_{cur_epoch:03d}.pt")
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, checkpoint_path))
    torch.save(save_obj, checkpoint_path)
    
    trainable_path = os.path.join(output_dir, f"stage1_trainable_epoch_{cur_epoch:03d}.pt")
    model_no_ddp.save_trainable_params(trainable_path)
    
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
    
    header = f'Train Epoch: [{epoch}]'
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
    
    return metric_logger.meters["loss"].global_avg


def evaluate_vqa(model, dataloader, device, max_samples=1000):
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, samples in enumerate(dataloader):
            if i * dataloader.batch_size >= max_samples:
                break
                
            samples['image'] = samples['image'].to(device)
            questions = samples['question']
            ground_truth = samples['answer']
            
            for j in range(len(questions)):
                image = samples['image'][j:j+1]
                question = questions[j]
                gt_answer = ground_truth[j]
                
                try:
                    generated_answer = model.generate(
                        image=image,
                        question=question,
                        max_length=20,
                        num_beams=1,
                        do_sample=False
                    )
                    
                    if generated_answer.lower().strip() == gt_answer.lower().strip():
                        correct += 1
                    
                except Exception as e:
                    print(f"Error generating answer: {e}")
                
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"VQA Evaluation: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    output_dir = args.out_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])
    else:
        sampler = None
        model = model.to(device)
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        sampler=sampler,
        shuffle=(sampler is None), 
        drop_last=True,
        num_workers=4
    )
    
    optimizer = set_optimizer(model, init_lr=2e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=8e-5,
        init_lr=2e-4,
        warmup_start_lr=1e-6,
        warmup_steps=2000,
    )
    
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        
        model.train()
        
        avg_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, epoch, args, device
        )
        
        if (epoch + 1) % args.save_freq == 0 or epoch == epochs - 1:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, output_dir)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(output_dir, "best_stage1_vqa.pt")
                if os.path.exists(checkpoint_path):
                    import shutil
                    shutil.copy2(checkpoint_path, best_path)
                    print(f"New best model saved with loss: {best_loss:.4f}")
        
        if args.eval_during_training and (epoch + 1) % 5 == 0:
            try:
                accuracy = evaluate_vqa(model, train_dataloader, device, max_samples=100)
                print(f"Epoch {epoch}: VQA Accuracy = {accuracy:.4f}")
            except Exception as e:
                print(f"Evaluation failed: {e}")
    
    return model


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starting Stage 1 VQA Training...')
    print(" # PID :", os.getpid())
    
    parser = argparse.ArgumentParser(description='Stage 1 VQA Training')
    
    parser.add_argument('--out_dir', default='', help='')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--bs', type=int, default=8, help='')
    parser.add_argument('--save_freq', type=int, default=1, help='')
    
    parser.add_argument('--data_root', default='', help='')
    parser.add_argument('--use_vqav2', action='store_true', default=True, help='')
    parser.add_argument('--max_samples', type=int, help='')
    
    parser.add_argument('--llama_model', default='', help='')
    parser.add_argument('--num_query_token', type=int, default=32, help='')
    parser.add_argument('--low_resource', action='store_true', help='')
    
    parser.add_argument('--device', default='cuda', help='')
    parser.add_argument('--distributed', action='store_true', default=True, help='')
    parser.add_argument('--amp', action='store_true', default=True, help='')
    parser.add_argument('--eval_during_training', action='store_true', help='')
    
    parser.add_argument('--dist_url', default="env://")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--disable_random_seed', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    print(f'Arguments: {vars(args)}')
    
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    
    if args.distributed:
        init_distributed_mode(args)
    
    print("Loading VQA dataset...")
    dataset = VQAEVCapDataset(
        data_root=args.data_root,
        split="train",
        max_samples=args.max_samples,
        use_vqav2=args.use_vqav2
    )
    
    print("Creating EVCap model...")
    model = AlignThenInject(
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=args.num_query_token,
        llama_model=args.llama_model,
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=128,
        end_sym='\n',
        low_resource=args.low_resource,
        device_8bit=0,
        stage="vqa"
    )
    
    print("Starting training...")
    trained_model = train(dataset, model, args)
    
    print("Stage 1 VQA training completed!")
    print(f"Model saved in: {args.out_dir}")
    print("Ready for Stage 2 caption training.")


if __name__ == '__main__':
    main()