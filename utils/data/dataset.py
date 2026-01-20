import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Optional


class VQADataset(Dataset):
    
    def __init__(
        self,
        data_root: str = "",
        split: str = "train",
        max_samples: Optional[int] = None,
        use_vqav2: bool = True,
    ):
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        self.use_vqav2 = use_vqav2
        
        if use_vqav2:
            self._load_vqav2_data()
        else:
            self._load_vqa_data()
        
        if max_samples and len(self.annotation) > max_samples:
            self.annotation = random.sample(self.annotation, max_samples)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        print(f"VQA Dataset loaded: {len(self.annotation)} samples ({split} split)")
    
    def _load_vqav2_data(self):
        if self.split == "train":
            question_file = os.path.join(self.data_root, "vqa_v2", "v2_OpenEnded_mscoco_train2014_questions.json")
            annotation_file = os.path.join(self.data_root, "vqa_v2", "v2_mscoco_train2014_annotations.json")
            self.image_root = os.path.join(self.data_root, "coco", "coco2014", "train2014")
        else:
            question_file = os.path.join(self.data_root, "vqa_v2", "v2_OpenEnded_mscoco_val2014_questions.json")
            annotation_file = os.path.join(self.data_root, "vqa_v2", "v2_mscoco_val2014_annotations.json")
            self.image_root = os.path.join(self.data_root, "coco", "val2014")
        
        with open(question_file, 'r') as f:
            questions_data = json.load(f)
        
        with open(annotation_file, 'r') as f:
            annotations_data = json.load(f)
        
        qid_to_answer = {}
        for ann in annotations_data['annotations']:
            answers = [a['answer'] for a in ann['answers']]
            most_common_answer = max(set(answers), key=answers.count)
            qid_to_answer[ann['question_id']] = most_common_answer
        
        self.annotation = []
        for q in questions_data['questions']:
            question_id = q['question_id']
            if question_id in qid_to_answer:
                self.annotation.append({
                    'question_id': question_id,
                    'image_id': q['image_id'],
                    'question': q['question'],
                    'answer': qid_to_answer[question_id]
                })
    
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        if self.use_vqav2:
            if self.split == "train":
                img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
            else:
                img_file = f'COCO_val2014_{int(ann["image_id"]):012d}.jpg'
        else:
            img_file = ann.get('image_file', f'image_{ann["image_id"]}.jpg')
        
        image_path = os.path.join(self.image_root, img_file)
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        return {
            "image": image,
            "question": ann["question"],
            "answer": ann["answer"],
            "question_id": ann["question_id"],
            "image_id": ann["image_id"]
        }


class COCOCaptionDataset(Dataset):
    
    def __init__(
        self,
        data_root: str = "",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.split = split
        
        if split == "train":
            ann_path = os.path.join(data_root, 'coco2014', 'annotations/captions_train2014.json')
            self.vis_root = os.path.join(data_root, 'coco2014', 'train2014')
        else:
            ann_path = os.path.join(data_root, 'coco2014', 'annotations/captions_val2014.json')
            self.vis_root = os.path.join(data_root, 'coco2014', 'val2014')
        
        self.annotation = []
        with open(ann_path, "r") as f:
            coco_data = json.load(f)
            self.annotation.extend(coco_data['annotations'])
        
        if max_samples and len(self.annotation) > max_samples:
            self.annotation = random.sample(self.annotation, max_samples)
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        print(f"COCO Caption Dataset loaded: {len(self.annotation)} samples ({split} split)")
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        if "train" in self.vis_root:
            img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        else:
            img_file = f'COCO_val2014_{int(ann["image_id"]):012d}.jpg'
        
        image_path = os.path.join(self.vis_root, img_file)
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        caption = ann["caption"]
        
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "caption_id": ann.get("id", index)
        }