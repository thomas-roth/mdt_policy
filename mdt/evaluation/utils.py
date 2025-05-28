from collections import defaultdict
import contextlib
import logging
import os
from pathlib import Path
from typing import Union
import importlib

import cv2
import hydra
import imageio
import numpy as np
from omegaconf import OmegaConf
import pyhash
import torch
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import wandb

from mdt.utils.utils import add_text, format_sftp_path


ROOT_OUTPUT_PATH = "/home/troth/code/hiwi/mdt_policy/outputs"
ENC_RESIZE_SHAPE = (224, 224) # size of encoder input images
DEC_SELF_RESIZE_SHAPE = (250, 250)
DEC_CROSS_RESIZE_SHAPE = (100, 250) # preserves 4:10 aspect ratio


hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_evaluation_checkpoint(cfg):
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    ).cuda()
    return pl_module


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*.hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pl_module_from_checkpoint(
    filepath: Union[Path, str],
    epoch: int = 1,
    overwrite_cfg: dict = {},
    use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
     # Load EMA weights if they exist and the flag is set
    if use_ema_weights:
        checkpoint_data = torch.load(ckpt_path)
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']
            
            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}
            
            model.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            print("Warning: No EMA weights found in checkpoint!")

    print(f"Finished loading model {ckpt_path}")
    return model



def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    
    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    )
    # model = Hulc.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def get_default_beso_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0, eval_cfg_overwrite={}):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    # GlobalHydra.instance().clear()
    # datasets_cfg = hydra.initialize("datamodule/datasets/vision_lang.yaml")
    # since we don't use the trainer during inference, manually set up data_module
    # cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    
    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
        use_ema_weights=True
    )
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    s = sum(demo_task_counter.values())
    success_rate = (sum(live_task_counter.values()) / s if s > 0 else 0) * 100
    logger.info(f"Average Success Rate {mod} = {success_rate:.0f}%")
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def gen_heatmaps(attns_sequences, merge_attn_heads=True, gen_for_enc=False, gen_for_dec_self=True, gen_for_dec_cross=True, save_gifs_not_jpgs=True):
    output_dirs = [os.path.join(ROOT_OUTPUT_PATH, day, time) for day in os.listdir(ROOT_OUTPUT_PATH) for time in os.listdir(Path(ROOT_OUTPUT_PATH) / day)]
    latest_output_dir = max(output_dirs)
    heatmap_output_path = f"{latest_output_dir}/attvis"

    input_tokens_enc = ["task", "cams0", "cams1", "cams2"] # 1 token for task instruction, 3 tokens for camera imgs (cannot be separated bc of cross-attn in PerceiverResampler)
    input_tokens_dec_self = [f"action{i}" for i in range(10)] # 10 tokens for prediction of next 10 actions

    heatmaps = [defaultdict(list) for _ in range(len(attns_sequences))]

    for sequence_number, attns_sequence in tqdm(enumerate(attns_sequences), total=len(attns_sequences), desc="Generating attn heatmaps for sequences"):
        for attns_task in tqdm(attns_sequence, leave=False):
            subtask = attns_task["subtask"].replace(" ", "_")

            for step_number, attns_step_plus_model_input in enumerate(attns_task["attns"]):
                attns_step = attns_step_plus_model_input["attns"]
                img_static = attns_step_plus_model_input["img_static"][0][0].cpu().detach().numpy().transpose(1, 2, 0) # (H, W, C) = (224, 224, 3)
                img_gripper = attns_step_plus_model_input["img_gripper"][0][0].cpu().detach().numpy().transpose(1, 2, 0) # (H, W, C) = (84, 84, 3)

                img_static = cv2.cvtColor(img_static, cv2.COLOR_RGB2GRAY)
                img_gripper = cv2.cvtColor(img_gripper, cv2.COLOR_RGB2GRAY)
                img_gripper_resized = cv2.resize(img_gripper, (img_static.shape[0], img_static.shape[1]), interpolation=cv2.INTER_NEAREST)

                if attns_step is None:
                    continue # skip if step action already predicted in previous step (multistep prediction) 
                
                if save_gifs_not_jpgs:
                    heatmap_gifs = defaultdict(lambda: defaultdict(list))

                for noise_level_inv, attns_noise_level in enumerate(attns_step):
                    noise_level = len(attns_step) - noise_level_inv - 1

                    attns_noise_level_enc = attns_noise_level["enc"]
                    attns_noise_level_dec = attns_noise_level["dec"]

                    if gen_for_enc:
                        for layer, attns_enc in enumerate(attns_noise_level_enc):
                            attns_enc = attns_enc.cpu().detach().numpy() # (B, nh, Te, Te) = (1, 8, 4, 4)

                            # normalize attention weights
                            attns_enc = (attns_enc - attns_enc.min()) / (attns_enc.max() - attns_enc.min())
                            attns_enc = (attns_enc * 255)
                            
                            if merge_attn_heads:
                                attns_enc = attns_enc[0].mean(axis=0).astype(np.uint8) # (Te, Te) = (4, 4)

                                attns_enc = cv2.resize(attns_enc, ENC_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                attns_enc = cv2.applyColorMap(attns_enc, cv2.COLORMAP_JET) # (H, W, C) = (224, 224, 3)
                                attns_enc = draw_token_labels_onto_heatmap(attns_enc, input_tokens_enc, input_tokens_enc)

                                if save_gifs_not_jpgs:
                                    heatmap_gifs[layer]["enc"].append(attns_enc)
                                else:
                                    img_name = f"encoder_noise_level_{noise_level}_layer_{layer}_merged_heads"

                                    heatmaps[sequence_number][subtask].append(wandb.Image(attns_enc, caption=img_name, masks={"img_static": {"mask_data": img_static}}))

                                    os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/enc", exist_ok=True)
                                    cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/enc/{img_name}.jpg", attns_enc)
                            else:
                                attns_enc = attns_enc[0].astype(np.uint8) # (nh, Te, Te) = (8, 4, 4)

                                for head_number, attns_enc_head in enumerate(attns_enc):
                                    attns_enc_head = cv2.resize(attns_enc_head, ENC_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                    attns_enc_head = cv2.applyColorMap(attns_enc_head, cv2.COLORMAP_JET) # (H, W, C) = (224, 224, 3)
                                    attns_enc_head = draw_token_labels_onto_heatmap(attns_enc_head, input_tokens_enc, input_tokens_enc)

                                    if save_gifs_not_jpgs:
                                        heatmap_gifs[layer]["enc"].append(attns_enc_head)
                                    else:
                                        img_name = f"encoder_noise_level_{noise_level}_layer_{layer}_head_{head_number}"
                                        
                                        heatmaps[sequence_number][subtask].append(wandb.Image(attns_enc_head, caption=img_name, masks={"img_static": {"mask_data": img_static}}))
                                        
                                        os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/enc", exist_ok=True)
                                        cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/enc/{img_name}.jpg", attns_enc_head)
                    
                    if gen_for_dec_self or gen_for_dec_cross:
                        for layer, attns_dec in enumerate(attns_noise_level_dec):
                            self_attns_dec = attns_dec["self"].cpu().detach().numpy() # (B, nh, Td, Td) = (1, 8, 10, 10)
                            cross_attns_dec = attns_dec["cross"].cpu().detach().numpy() # (B, nh, Td, Te) = (1, 8, 10, 4)

                            # normalize attention weights
                            self_attns_dec = (self_attns_dec - self_attns_dec.min()) / (self_attns_dec.max() - self_attns_dec.min())
                            self_attns_dec = (self_attns_dec * 255)
                            cross_attns_dec = (cross_attns_dec - cross_attns_dec.min()) / (cross_attns_dec.max() - cross_attns_dec.min())
                            cross_attns_dec = (cross_attns_dec * 255)

                            if merge_attn_heads:
                                if gen_for_dec_self:
                                    self_attns_dec = self_attns_dec[0].mean(axis=0).astype(np.uint8) # (Td, Td) = (10, 10)

                                    self_attns_dec = cv2.resize(self_attns_dec, DEC_SELF_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                    self_attns_dec = cv2.applyColorMap(self_attns_dec, cv2.COLORMAP_JET) # (H, W, C) = (250, 250, 3)
                                    self_attns_dec = draw_token_labels_onto_heatmap(self_attns_dec, input_tokens_dec_self, input_tokens_dec_self)

                                    if save_gifs_not_jpgs:
                                        self_attns_dec_rgb = cv2.cvtColor(self_attns_dec, cv2.COLOR_BGR2RGB)
                                        heatmap_gifs[layer]["dec_self"].append(self_attns_dec_rgb)
                                    else:
                                        img_name = f"decoder_self_attn_noise_level_{noise_level}_layer_{layer}_merged_heads"

                                        heatmaps[sequence_number][subtask].append(wandb.Image(self_attns_dec, caption=img_name)) # mask: encoder output ("context" in mdtv_transformer.forward())
                                        
                                        os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_self", exist_ok=True)
                                        cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_self/{img_name}.jpg", self_attns_dec)
                                if gen_for_dec_cross:
                                    cross_attns_dec = cross_attns_dec[0].mean(axis=0).astype(np.uint8) # (Td, Te) = (10, 4)

                                    cross_attns_dec = cv2.resize(cross_attns_dec, DEC_CROSS_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                    cross_attns_dec = cv2.applyColorMap(cross_attns_dec, cv2.COLORMAP_JET) # (H, W, C) = (250, 100, 3)
                                    cross_attns_dec = draw_token_labels_onto_heatmap(cross_attns_dec, input_tokens_enc, input_tokens_dec_self)

                                    if save_gifs_not_jpgs:
                                        cross_attns_dec_rgb = cv2.cvtColor(cross_attns_dec, cv2.COLOR_BGR2RGB)
                                        heatmap_gifs[layer]["dec_cross"].append(cross_attns_dec_rgb)
                                    else:
                                        img_name = f"decoder_cross_attn_noise_level_{noise_level}_layer_{layer}_merged_heads"

                                        heatmaps[sequence_number][subtask].append(wandb.Image(cross_attns_dec, caption=img_name, masks={"img_static": {"mask_data": img_static},
                                                                                                                                            "img_gripper": {"mask_data": img_gripper_resized}}))
                                        
                                        os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_cross", exist_ok=True)
                                        cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_cross/{img_name}.jpg", cross_attns_dec)
                            else:
                                self_attns_dec = self_attns_dec[0].astype(np.uint8) # (nh, Td, Td) = (8, 10, 10)
                                cross_attns_dec = cross_attns_dec[0].astype(np.uint8) # (nh, Td, Te) = (8, 10, 4)

                                for head_number, (self_attns_dec_head, cross_attns_dec_head) in enumerate(zip(self_attns_dec, cross_attns_dec)):
                                    if gen_for_dec_self:
                                        self_attns_dec_head = cv2.resize(self_attns_dec, DEC_SELF_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                        self_attns_dec_head = cv2.applyColorMap(self_attns_dec_head, cv2.COLORMAP_JET) # (H, W, C) = (250, 250, 3)
                                        self_attns_dec_head = draw_token_labels_onto_heatmap(self_attns_dec_head, input_tokens_dec_self, input_tokens_dec_self)

                                        if save_gifs_not_jpgs:
                                            self_attns_dec_head_rgb = cv2.cvtColor(self_attns_dec_head, cv2.COLOR_BGR2RGB)
                                            heatmap_gifs[layer]["dec_self"].append(self_attns_dec_head_rgb)
                                        else:
                                            img_name = f"decoder_self_attn_noise_level_{noise_level}_layer_{layer}_head_{head_number}"

                                            heatmaps[sequence_number][subtask].append(wandb.Image(self_attns_dec_head, caption=img_name))

                                            os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_self", exist_ok=True)
                                            cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_self/{img_name}.jpg", self_attns_dec_head)
                                    
                                    if gen_for_dec_cross:
                                        cross_attns_dec_head = cv2.resize(cross_attns_dec_head, DEC_CROSS_RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
                                        cross_attns_dec_head = cv2.applyColorMap(cross_attns_dec_head, cv2.COLORMAP_JET) # (H, W, C) = (250, 100, 3)
                                        cross_attns_dec_head = draw_token_labels_onto_heatmap(cross_attns_dec_head, input_tokens_enc, input_tokens_dec_self)

                                        if save_gifs_not_jpgs:
                                            cross_attns_dec_head_rgb = cv2.cvtColor(cross_attns_dec_head, cv2.COLOR_BGR2RGB)
                                            heatmap_gifs[layer]["dec_cross"].append(cross_attns_dec_head_rgb)
                                        else:
                                            img_name = f"decoder_cross_attn_noise_level_{noise_level}_layer_{layer}_head_{head_number}"

                                            heatmaps[sequence_number][subtask].append(wandb.Image(cross_attns_dec_head, caption=img_name, masks={"img_static": {"mask_data": img_static},
                                                                                                                                                        "img_gripper": {"mask_data": img_gripper_resized}}))
                                            
                                            os.makedirs(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_cross", exist_ok=True)
                                            cv2.imwrite(f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/layer_{layer}/dec_cross/{img_name}.jpg", cross_attns_dec_head)

                if save_gifs_not_jpgs:
                    for layer, gif_frames in heatmap_gifs.items():
                        for key in ["enc", "dec_self", "dec_cross"]:
                            if key in gif_frames and len(gif_frames[key]) > 0:
                                # reverse frame order for enc gifs as noise level increases instead of decreases
                                if key == "enc":
                                    gif_frames[key].reverse()

                                gif_path = f"{heatmap_output_path}/seq_{sequence_number}/step_{step_number}/"
                                os.makedirs(gif_path, exist_ok=True)

                                imageio.mimsave(f"{gif_path}/layer_{layer}_{key}.gif", gif_frames[key], duration=0.75, loop=0)

                                heatmaps[sequence_number][subtask].append(
                                    wandb.Video(f"{gif_path}/layer_{layer}_{key}.gif", caption=f"step-{step_number}_layer-{layer}_{key.replace('_', '-')}-attn", format="gif")
                                )


    for sequence_number, heatmaps_sequence in enumerate(heatmaps):
        num_zeros_heatmaps = max(len(str(len(heatmaps_sequence[subtask]))) for subtask in heatmaps_sequence.keys())
        num_zeros_seqs = max(len(str(sequence_number)) for sequence_number in range(len(heatmaps)))
        for subtask in heatmaps_sequence.keys():
            if save_gifs_not_jpgs:
                print(f"{len(heatmaps[sequence_number][subtask]):{num_zeros_heatmaps}} heatmap gifs for sequence {sequence_number:0{num_zeros_seqs}} and subtask {subtask}")
            else:
                print(f"{len(heatmaps[sequence_number][subtask]):{num_zeros_heatmaps}} heatmaps for sequence {sequence_number:0{num_zeros_seqs}} and subtask {subtask}")
    
    return heatmaps


def draw_token_labels_onto_heatmap(heatmap, x_labels, y_labels):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    x_cell_height = max(cv2.getTextSize(x_label, font_face, font_scale, thickness)[0][0] for x_label in x_labels) # dynamic depending on longest x label
    y_cell_width = max(cv2.getTextSize(y_label, font_face, font_scale, thickness)[0][0] for y_label in y_labels) # dynamic depending on longest y label
    margin_heatmap_labels = 5 # no. of pixels between heatmap and labels
    
    heatmap_height, heatmap_width = heatmap.shape[:2]

    heatmap_canvas = np.ones((heatmap_height + x_cell_height + margin_heatmap_labels, heatmap_width + y_cell_width + margin_heatmap_labels, 3), dtype=np.uint8) * 255
    heatmap_canvas[:heatmap_height, -heatmap_width:] = heatmap # paste heatmap onto top right of canvas

    heatmap_canvas_rotated = cv2.rotate(heatmap_canvas, cv2.ROTATE_90_CLOCKWISE) # x labels are written rotated s.t. they fit onto canvas

    x_cell_width = heatmap_width // len(x_labels)
    x_cell_middle = x_cell_width // 2 + 5
    for i, label in enumerate(x_labels):
        x = max(0, x_cell_height - cv2.getTextSize(label, font_face, font_scale, thickness)[0][0]) # right align text w/ overflow protection
        y = y_cell_width + margin_heatmap_labels + i * x_cell_width + x_cell_middle
        cv2.putText(heatmap_canvas_rotated, label, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)

    heatmap_canvas = cv2.rotate(heatmap_canvas_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    y_cell_height = heatmap_height // len(y_labels)
    y_cell_middle = y_cell_height // 2 + 5
    for i, label in enumerate(y_labels):
        x = 0
        y = i * y_cell_height + y_cell_middle
        cv2.putText(heatmap_canvas, label, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    return heatmap_canvas
