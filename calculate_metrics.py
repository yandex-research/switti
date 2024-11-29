import os

import ImageReward
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from PIL import Image
from tqdm.auto import tqdm
import dist


@torch.no_grad()
def calc_pick_or_clip_scores(model, image_inputs, text_inputs, batch_size=50):
    assert len(image_inputs) == len(text_inputs["input_ids"])
    assert len(text_inputs.keys()) == 2

    scores = torch.zeros(len(image_inputs))
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = {
            "input_ids": text_inputs["input_ids"][i : i + batch_size],
            "attention_mask": text_inputs["attention_mask"][i : i + batch_size],
        }
        # embed
        with torch.cuda.amp.autocast():
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            text_embs = model.get_text_features(**text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i : i + batch_size] = (text_embs * image_embs).sum(-1)
    return scores.cpu()


@torch.no_grad()
def calculate_image_reward_score(
    images,
    prompts,
    device="cuda",
    batch_size=50,
    image_reward_path="ImageReward-v1.0",
):
    model = ImageReward.load(image_reward_path, device=device).eval()

    scores = []
    for i in range(0, len(prompts), batch_size):
        # text encode
        with torch.amp.autocast("cuda"):
            text_input = model.blip.tokenizer(
                prompts[i: i + batch_size],
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(device)

            processed_images = torch.stack(
                [
                    model.preprocess(image).to(device)
                    for image in images[i: i + batch_size]
                ]
            )
            image_embeds = model.blip.visual_encoder(processed_images)

            # text encode cross attention with image
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(device)
            text_output = model.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        txt_features = text_output.last_hidden_state[:, 0].float()  # (feature_dim)
        rewards = model.mlp(txt_features)
        rewards = (rewards - model.mean) / model.std

        scores.extend(rewards[:, 0].tolist())

    return np.mean(scores)


@torch.no_grad()
def calculate_scores(
    images,
    prompts,
    device="cuda",
    clip_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    pickscore_model_name_or_path="yuvalkirstain/PickScore_v1",
    image_reward_path=None,
):
    processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(clip_model_name_or_path).eval().to(device)
    pickscore_model = (
        AutoModel.from_pretrained(pickscore_model_name_or_path).eval().to(device)
    )

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )[
        "pixel_values"
    ].to(device)

    text_inputs = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    print("Evaluating PickScore...")
    pick_score = calc_pick_or_clip_scores(
        pickscore_model, image_inputs, text_inputs
    ).mean()

    print("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_or_clip_scores(
        clip_model, image_inputs, text_inputs
    ).mean()

    print("Evaluating ImageReward...")
    image_reward = calculate_image_reward_score(
        images,
        prompts,
        device,
        image_reward_path=image_reward_path,
    )
    image_reward = torch.full_like(clip_score, image_reward)

    return pick_score, clip_score, image_reward


@torch.no_grad()
def distributed_metrics_with_csv(
    pipe,
    csv_path,
    args,
):
    pipe.switti.eval()
    max_count = args.metrics_max_count
    rank_batches, *_ = prepare_prompts(csv_path, args.eval_batch_size, max_count)
    assert max_count % (args.eval_batch_size * dist.get_world_size()) == 0
    local_images, local_prompts = [], []
    for batch in tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0)):
        texts = [str(prompt) for prompt in batch
                 for _ in range(args.num_images_for_metrics)]
        image_tensors = pipe(
            prompt=texts,
            seed=args.seed,
            cfg=args.guidance,
            top_k=args.top_k,
            top_p=args.top_p,
            more_smooth=False,
            return_pil=False,
        )

        local_images.extend(image_tensors)
        local_prompts.extend(texts)

    local_images = torch.stack(local_images).cuda()

    local_pick_score, local_clip_score, local_image_reward = calculate_scores(
        [to_PIL_image(image) for image in local_images.clone()],
        local_prompts,
        device=dist.get_device(),
        clip_model_name_or_path=args.clip_model_name_or_path,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        image_reward_path=args.image_reward_path,
    )
    # Done.
    dist.barrier()
    return local_images, local_pick_score, local_clip_score, local_image_reward


def save_images(images, prompts, save_path):
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{i:04d}.jpg"))
    if prompts:
        with open(os.path.join(save_path, "prompts.txt"), "w") as f:
            f.writelines("\n".join(prompts))


def prepare_prompts(prompts_path, batch_size=1, max_count=None):
    assert max_count % dist.get_world_size() == 0
    df = pd.read_csv(prompts_path)
    all_text = list(df["captions"])

    if max_count is not None:
        all_text = all_text[:max_count]

    num_batches = (
        (len(all_text) - 1) // (batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


def to_PIL_image(image_tensor):
    # [c, h, w] -> [h, w, c]
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor)
    img = (image_tensor.permute(1, 2, 0) * 255).cpu().numpy()
    return Image.fromarray(img.astype(np.uint8))
