import logging
import os
import warnings
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

import random
import pickle
import faiss
import re

from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from models.Qformer_txt import BertLMHeadModel_txt as QFormerTxtModel
from transformers import LlamaTokenizer
from transformers import BertTokenizer


class AlignThenInject(Blip2Base):
    def __init__(
        self,
        ext_path=None,
        vit_model="eva_clip_g",
        q_former_model="",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=160,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
        stage="caption",
    ):
        super().__init__()

        self.low_resource = low_resource
        self.stage = stage
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.topn = topn
        self.num_query_token_txt = num_query_token_txt

        print(f'Initializing AlignThenInject - Stage: {stage}')
        if stage == "vqa":
            print(f'  Stage 1 (Align): Visual-Language Alignment')
        else:
            print(f'  Stage 2 (Inject): Knowledge Injection')

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer, keep query_tokens trainable")
        print('Loading Q-Former Done')

        if stage == "caption":
            print('Loading Q-Former-txt')
            self.bert_tokenizer = self.init_tokenizer()
            self.Qformer_txt, self.query_tokens_txt = self.init_Qformer_txt(
                num_query_token_txt, self.Qformer.config.hidden_size
            )
            self.Qformer_txt.resize_token_embeddings(len(self.bert_tokenizer))
            self.Qformer_txt.cls = None
            self.load_from_pretrained(url_or_filename=q_former_model)
            
            if freeze_qformer:
                for name, param in self.Qformer_txt.named_parameters():
                    param.requires_grad = False
                self.Qformer_txt = self.Qformer_txt.eval()
                self.Qformer_txt.train = disabled_train
                self.query_tokens_txt.requires_grad = True
                logging.info("freeze Qformer_txt, keep query_tokens_txt trainable")
            print('Loading Q-Former-txt Done')
        else:
            self.Qformer_txt = None
            self.query_tokens_txt = None
            self.bert_tokenizer = None

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        else:
            self.prompt_list = []

        if stage == "caption" and ext_path:
            with open(ext_path, 'rb') as f:
                ext_base_img, self.ext_base_img_id = pickle.load(f)
                feature_library_cpu = ext_base_img.cpu().numpy()
                faiss.normalize_L2(feature_library_cpu)
                self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
                self.feat_index.add(feature_library_cpu)
        else:
            self.feat_index = None
            self.ext_base_img_id = None

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def pre_name(self, caption):
        caption = re.sub(r"([_!,'\"()*#:;~])", " ", caption.lower())
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n").strip(" ")
        return caption

    def retrieve_similar_features(self, query_features, feat_index, image_id, top_k=5, sub_top_k=32):
        if feat_index is None:
            return [[] for _ in range(query_features.shape[0])]
            
        batch_size, nums, dims = query_features.shape
        query_features = query_features.view(-1, dims)   

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device=query_features.device)
        top_k_similarities = torch.tensor(top_k_similarities).to(device=query_features.device)
        top_k_similarities = top_k_similarities.view(batch_size, -1)

        indices = top_k_indices.view(batch_size, -1)

        re_txt_list_all = []    
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list: 
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list)
         
        sorted_batched_ret = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [self.pre_name(listB[idx]) for idx in indices]
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret.append(sorted_listB)
        return sorted_batched_ret

    def _resolve_cross_attention_module(self, preferred_idx: int = 0):
        encoder_layers = getattr(self.Qformer_txt.bert.encoder, "layer", None)
        if encoder_layers is None or len(encoder_layers) == 0:
            raise RuntimeError("Qformer_txt encoder not initialized properly.")

        preferred_idx = max(0, min(preferred_idx, len(encoder_layers) - 1))
        search_order = [preferred_idx] + [
            i for i in range(len(encoder_layers)) if i != preferred_idx
        ]

        for idx in search_order:
            layer = encoder_layers[idx]
            cross_attn = getattr(layer, "crossattention", None)
            if cross_attn is not None:
                if idx != preferred_idx:
                    warnings.warn(
                        f"Layer {preferred_idx} missing cross-attention, using layer {idx}.",
                        RuntimeWarning,
                    )
                return cross_attn.self

        warnings.warn(
            "No cross-attention layer found, falling back to self-attention.",
            RuntimeWarning,
        )
        return encoder_layers[preferred_idx].attention.self

    def compute_chamfer_similarity(self, img_feats, text_feats, text_mask):
        cross_attn_layer = self._resolve_cross_attention_module(preferred_idx=0)

        with torch.no_grad():
            proj_text = cross_attn_layer.query(text_feats)
            proj_img = cross_attn_layer.key(img_feats)

        proj_text = F.normalize(proj_text, p=2, dim=-1)
        proj_img = F.normalize(proj_img, p=2, dim=-1)
        text_mask = text_mask.float()

        scores = []
        for i in range(proj_img.size(0)):
            sim_matrix = torch.einsum('nld,qd->nlq', proj_text, proj_img[i])
            max_sim = sim_matrix.max(dim=2).values
            masked = max_sim * text_mask
            valid = text_mask.sum(dim=1).clamp(min=1e-6)
            avg_scores = masked.sum(dim=1) / valid
            scores.append(avg_scores)

        return torch.stack(scores, dim=0)

    def semantic_adaptive_entity_rectification(self, entity_candidates, query_image_features):
        filtered_candidates = []
        batch_avg_similarities = []
        batch_size = len(entity_candidates)

        all_entities_flat = [entity for entities in entity_candidates for entity in entities]

        if not all_entities_flat:
            zeros = torch.zeros(batch_size, device=query_image_features.device)
            return [[] for _ in range(batch_size)], zeros

        text_tokens = self.bert_tokenizer(
            all_entities_flat,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=80
        ).to(query_image_features.device)

        with torch.no_grad():
            text_embeds = self.Qformer_txt.bert.embeddings(
                input_ids=text_tokens.input_ids
            )

        scores_matrix = self.compute_chamfer_similarity(
            query_image_features,
            text_embeds,
            text_tokens.attention_mask
        )
        start_idx = 0
        for b_idx in range(batch_size):
            current_entities = entity_candidates[b_idx]
            num_entities = len(current_entities)

            if num_entities == 0:
                filtered_candidates.append([])
                batch_avg_similarities.append(0.0)
                continue

            current_scores = scores_matrix[b_idx, start_idx:start_idx + num_entities]
            mean_score = current_scores.mean()
            std_score = torch.nan_to_num(
                current_scores.std(unbiased=False), nan=0.0
            )
            threshold = mean_score + std_score
            threshold_val = threshold.item()

            entity_score_pairs = sorted(
                zip(current_entities, current_scores.tolist()),
                key=lambda x: x[1],
                reverse=True
            )

            filtered_list = []
            filtered_scores = []
            for entity, score in entity_score_pairs:
                if score >= threshold_val:
                    filtered_list.append(entity)
                    filtered_scores.append(score)

            filtered_candidates.append(filtered_list)
            avg_sim = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0.0
            batch_avg_similarities.append(avg_sim)

            start_idx += num_entities

        return filtered_candidates, torch.tensor(batch_avg_similarities, device=query_image_features.device)
        
    def visual_anchored_semantic_reconstruction(self, img_features, txt_features, entity_similarities=None):
        batch_size = img_features.shape[0]
        attention_scores = torch.bmm(txt_features, img_features.transpose(1, 2))
        attention_probs = F.softmax(attention_scores / (img_features.size(-1) ** 0.5), dim=-1)
        
        txt_context_from_img = torch.bmm(attention_probs, img_features)
        
        sim_gates = entity_similarities.view(batch_size, 1, 1)
        alpha = 0.35 * sim_gates

        txt_features_enhanced = txt_features + alpha * txt_context_from_img
        
        return img_features, txt_features_enhanced

    def contextual_confidence_calibration(self, img_features, txt_features, entity_similarities=None):
        batch_size = img_features.shape[0]
        
        img_global = torch.mean(img_features, dim=1, keepdim=True)

        sim_gates = entity_similarities.view(batch_size, 1, 1)
        
        txt_consistency_scores = torch.matmul(
            F.normalize(txt_features, p=2, dim=-1),
            F.normalize(img_global, p=2, dim=-1).transpose(1, 2)
        )
        
        consistency_gate = torch.sigmoid(txt_consistency_scores * 5)
        final_txt_gate = consistency_gate * (0.5 + 0.5 * sim_gates)
        
        txt_features_gated = txt_features * final_txt_gate
        
        return img_features, txt_features_gated

    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                
                wrapped_embed_i = torch.cat([p_before_embeds, img_embeds_i, p_after_embeds], dim=1)
                emb_lists.append(wrapped_embed_i)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(
                torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
                
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state

            if self.stage == "vqa":
                img_feat_mean = torch.mean(query_output_img, dim=1, keepdim=True)
                placeholder = img_feat_mean.expand(-1, self.num_query_token_txt, -1)
                query_output_all = torch.cat([query_output_img, placeholder], dim=1)
                
            else:
                if self.feat_index is not None:
                    re_txt_list_all = self.retrieve_similar_features(
                        query_output_img, self.feat_index, self.ext_base_img_id, 
                        top_k=self.topn, sub_top_k=32
                    )
                    
                    re_txt_list_deduplicated = []  
                    for sublist in re_txt_list_all:
                        sublist_new = []
                        for item in sublist:
                            if item not in sublist_new:
                                sublist_new.append(item)
                                if len(sublist_new) > self.topn: 
                                    break
                        re_txt_list_deduplicated.append(sublist_new)

                    filtered_entities, entity_avg_similarities = self.semantic_adaptive_entity_rectification(
                        re_txt_list_deduplicated, query_output_img)

                    re_txt_list_batch = []
                    for sublist in filtered_entities:
                        joined_text = " [SEP] ".join(sublist)
                        re_txt_list_batch.append(joined_text)

                    text = self.bert_tokenizer(
                        re_txt_list_batch,
                        truncation=True,
                        padding="longest",
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(image.device)

                    query_tokens_txt = self.query_tokens_txt.expand(image_embeds.shape[0], -1, -1)
                    query_atts_txt = torch.ones(query_tokens_txt.size()[:-1], dtype=torch.long).to(device)

                    query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device)
                    query_output_img_atts = torch.cat([query_atts_txt, query_output_img_atts], dim=1)

                    attention_mask = text.attention_mask
                    query_outputs_txt = self.Qformer_txt.bert(
                        text.input_ids,
                        query_embeds=query_tokens_txt,
                        attention_mask=attention_mask,
                        encoder_hidden_states=query_output_img,
                        encoder_attention_mask=query_output_img_atts,
                        return_dict=True,
                    )
                    query_output_txt = query_outputs_txt.last_hidden_state[:, :query_tokens_txt.size(1), :]

                    query_output_img_compensated, query_output_txt_compensated = self.visual_anchored_semantic_reconstruction(
                        query_output_img, query_output_txt, entity_avg_similarities)

                    query_output_img_gated, query_output_txt_gated = self.contextual_confidence_calibration(
                        query_output_img_compensated, query_output_txt_compensated, entity_avg_similarities)

                    query_output_all = torch.cat([query_output_img_gated, query_output_txt_gated], dim=1)
                else:
                    img_feat_mean = torch.mean(query_output_img, dim=1, keepdim=True)
                    placeholder = img_feat_mean.expand(-1, self.num_query_token_txt, -1)
                    query_output_all = torch.cat([query_output_img, placeholder], dim=1)

            qform_all_proj = self.llama_proj(query_output_all)
            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long).to(device)
        
        return qform_all_proj, atts_qform_all_proj

    def forward(self, samples):
        if self.stage == "vqa":
            return self.forward_vqa(samples)
        else:
            return self.forward_caption(samples)
    
    def forward_vqa(self, samples):
        image = samples["image"]
        questions = samples["question"]
        answers = samples["answer"]
        
        qform_all_proj, atts_qform_all_proj = self.encode_img(image)
        
        batch_prompts = []
        for q, a in zip(questions, answers):
            prompt = f"Q: {q} A: {a}{self.end_sym}"
            batch_prompts.append(prompt)
        
        self.llama_tokenizer.padding_side = "right"
        text_tokens = self.llama_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        bos = torch.ones([qform_all_proj.shape[0], 1],
                        dtype=text_tokens.input_ids.dtype,
                        device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        
        empty_targets = torch.ones([qform_all_proj.shape[0], qform_all_proj.shape[1] + 1], 
                                   dtype=torch.long).to(image.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, qform_all_proj, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_qform_all_proj, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    def forward_caption(self, samples):
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(image)

        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list)
        else:
            prompt_embeds, atts_prompt = qform_all_proj, atts_qform_all_proj

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        bos = torch.ones([qform_all_proj.shape[0], 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], 
                       dtype=torch.long).to(image.device).fill_(-100)  
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        
        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_prompt, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    def set_stage(self, stage):
        self.stage = stage
        print(f"Model stage set to: {stage}")
        
        if stage == "caption" and hasattr(self, 'query_tokens_txt') and self.query_tokens_txt is not None:
            self.query_tokens_txt.requires_grad = True

    def get_trainable_params(self):
        trainable_params = []
        
        trainable_params.append(self.query_tokens)
        
        if self.stage == "caption" and hasattr(self, 'query_tokens_txt') and self.query_tokens_txt is not None:
            trainable_params.append(self.query_tokens_txt)
        
        for param in self.llama_proj.parameters():
            trainable_params.append(param)
        
        return trainable_params

    def load_stage1_weights(self, stage1_checkpoint_path):
        if not os.path.exists(stage1_checkpoint_path):
            print(f"Warning: Stage 1 checkpoint not found: {stage1_checkpoint_path}")
            return False
        
        try:
            print(f"Loading Stage 1 weights from: {stage1_checkpoint_path}")
            
            checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
            stage1_state_dict = checkpoint.get('model', {})
            
            current_state_dict = self.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for key in stage1_state_dict:
                if 'llama_proj' in key:
                    if key in current_state_dict:
                        if current_state_dict[key].shape == stage1_state_dict[key].shape:
                            current_state_dict[key] = stage1_state_dict[key]
                            loaded_keys.append(key)
                elif 'query_tokens' in key:
                    skipped_keys.append(key)
            
            self.load_state_dict(current_state_dict, strict=False)
            
            print(f"Loaded {len(loaded_keys)} parameters, skipped {len(skipped_keys)} parameters")
            
            return True
            
        except Exception as e:
            print(f"Error loading Stage 1 weights: {e}")
            import traceback
            traceback.print_exc()
            return False