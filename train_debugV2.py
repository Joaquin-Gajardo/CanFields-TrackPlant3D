#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime

import GPUtil

from multi_tools import *
from utils import wandb_name_init


def set_most_idle_gpu_as_visible():
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found!")
        return
    most_idle_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_idle_gpu.id)


set_most_idle_gpu_as_visible()


def velocity_test():
    args = build_args()

    # Seed: explicit --seed wins, otherwise 3407 + run_idx so that repeated runs
    # (--run_idx / --n_runs) automatically use different seeds. This gives a
    # meaningful mean +/- std over runs for every method (deterministic baselines
    # included), instead of identical repeated runs under a single fixed seed.
    seed = args.seed if args.seed is not None else 3407 + (args.run_idx or 0)
    setup_seed(seed)
    logging.info(f"Using random seed: {seed}")

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    logging.debug("running " + args.base_results_dir)
    signal.signal(signal.SIGINT, signal_handler)
    if not os.path.isfile(args.specifications):
        raise Exception("The experiment directory does not include such files")

    #######################################################
    specs = json.load(open(args.specifications))
    if args.dataset_name == "animals":
        anima_path = "datasets/Deforming4D/notexture/"
        specs["ShapePath"] = anima_path + args.object_name
    elif args.dataset_name == "defaust":
        defaust_path = "datasets/D_Faust/"
        specs["ShapePath"] = defaust_path + args.object_name

    elif args.dataset_name == "medical":
        medical_path = "datasets/Medical/"
        specs["ShapePath"] = medical_path + args.object_name
    elif args.dataset_name == "pheno4d":
        # For Pheno4D, use the object_name as the full path
        specs["ShapePath"] = args.object_name
    else:
        raise ValueError(
            f"The variable must be in either animals, medical, defaust, or pheno4d"
        )

    # Auto-construct split file path for pheno4d if not provided
    split_file = args.point_split_file
    if args.dataset_name == "pheno4d" and split_file is None:
        # Extract sequence name from path
        sequence_name = args.object_name.split("/")[-1]
        split_file = f"configs/point_splits/{sequence_name}_split.json"
        if os.path.exists(split_file):
            logging.info(f"Auto-detected point split file: {split_file}")
        else:
            logging.warning(f"Point split file not found: {split_file}")
            logging.warning("Training will use all points without train/test splitting")
            split_file = None

    # load_dataset
    if args.adjust_use == "True":
        ajust_weights = args.adjust_weights.split(",")
        ajust_coes = [
            float(ajust_weights[0]),
            float(ajust_weights[1]),
            float(ajust_weights[2]),
        ]
        sdf_dataset, obj_name = load_dataset(args, specs, ajust_coes, split_file=split_file, use_train_points=True)
    else:
        sdf_dataset, obj_name = load_dataset(args, specs, split_file=split_file, use_train_points=True)

    ############### Logging and Wandb ########################
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract base method name (remove _woAdaptor suffix if present) for directory naming and wandb
    base_method = args.methods.replace("_woAdaptor", "")

    # Optional run tag for repeated runs of the same sequence (e.g. "run2_")
    # Kept right after the timestamp so the experiment name stays "{timestamp}_run{idx}_{obj_name}".
    run_tag = f"run{args.run_idx}_" if getattr(args, "run_idx", None) else ""

    # Extract crop and sequence name from object_name path
    # For pheno4d: Pheno4D/processed/v2/organ_sequences/tomato/name -> crop=tomato, name=name
    # For others: use object_name as-is for backwards compatibility
    if args.dataset_name == "pheno4d":
        path_parts = args.object_name.split("/")
        crop = path_parts[-2]  # Extract crop (e.g., "tomato")
        sequence_name = path_parts[-1]  # Extract sequence name
        object_name = sequence_name  # For wandb logging
        # Build path: results/pheno4d/{method}/{crop}/{sequence_name}/{timestamp}_{experiment_name}/
        timestamped_exp_name = f"{timestamp}_{run_tag}{obj_name}"
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            base_method,  # Use base method name (without _woAdaptor suffix)
            crop,
            sequence_name,
            timestamped_exp_name,
        )
    else:
        # Original structure for other datasets
        object_name = args.object_name.split("/")[-1]
        timestamped_exp_name = f"{timestamp}_{run_tag}{obj_name}"
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            object_name,
            base_method,  # Use base method name (without _woAdaptor suffix)
            timestamped_exp_name,
        )
    os.makedirs(experiment_directory, exist_ok=True)

    # Prepare config for wandb
    wandb_config = {
        "dataset_name": args.dataset_name,
        "object_name": object_name,
        "obj_name": obj_name,
        "method": base_method,  # Base method name without suffix
        "use_adaptor": args.adjust_use == "True",  # Boolean flag
        "adjust_weights": args.adjust_weights if args.adjust_use == "True" else None,
        "clip": args.clip,
        "epoch_velo": args.epoch_velo,
        "pts_num": args.pts_num,
        "experiment_directory": experiment_directory,
        "timestamp": timestamp,
        "seed": seed,
        "run_idx": args.run_idx,
    }

    wandb = wandb_name_init(
        debug=False,
        wandb_dir=experiment_directory,
        config=wandb_config,
        timestamp=timestamp,
    )
    ##############################################################################

    start_idx = 0  #
    end_idx = len(sdf_dataset) - 1

    # load_network_latents
    if args.pts_num != None:
        save_pts_name = args.pts_num
    else:
        save_pts_name = "1"

    if args.adjust_use == "True":
        decoder, ajust, two_codes = load_network_model(args, specs, start_idx, end_idx)
        ajust, decoder, two_codes = load_model_latents(
            ajust, decoder, two_codes, experiment_directory, save_pts_name
        )

    else:
        if args.methods != None:
            decoder, start_end_vecs = load_network_model_others(
                args, specs, start_idx, end_idx
            )
        else:
            decoder, start_end_vecs = load_network_model(
                args, specs, start_idx, end_idx
            )
        decoder, start_end_vecs = load_model_latents_others(
            decoder, start_end_vecs, experiment_directory, save_pts_name
        )

    start_epoch = int(save_pts_name)
    num_epochs = specs["NumEpochs"]  # + 10_000
    # print('total_num_epochs: ',num_epochs)
    train_lens = len(sdf_dataset)
    logging.info("There are {} time instances".format(train_lens))
    logging.info("starting from epoch {}".format(start_epoch))
    ####################Load Model Params#####################
    decoder.train()
    times = []
    print("Use Other Methods? ", args.methods)
    if args.adjust_use == "True":
        trainer = Trainer(
            args,
            specs,
            ajust,
            decoder,
            sdf_dataset,
            two_codes,
            start_idx,
            end_idx,
            ajust_coes,
            timestamped_exp_name,
            experiment_directory,
        )
    else:
        trainer = Trainer_woadjust(
            args,
            specs,
            decoder,
            sdf_dataset,
            start_end_vecs,
            start_idx,
            end_idx,
            timestamped_exp_name,
            experiment_directory,
        )

    for epoch in range(start_epoch, num_epochs + 1):
        t1 = time.time()
        log_dict = trainer.forward_epoch(epoch)
        t2 = time.time()

        times.append(t2 - t1)
        wandb.log(log_dict)
        if epoch % 100 == 0:
            mean_time = np.array(times).mean()
            logging.info("computed_time_per100 = {}".format(100 * mean_time))
            times = []
        if epoch % 5000 == 0 or epoch == num_epochs - 1:  # 200
            trainer.save_model(epoch, experiment_directory)

        # Compute and save metrics periodically (every 5000 epochs or at the end)
        if epoch % 5000 == 0 or epoch == num_epochs - 1:
            try:
                metrics_dict = trainer.compute_and_save_metrics(epoch)
                if metrics_dict:
                    wandb.log(
                        {
                            "eval/mean_chamfer_distance": metrics_dict[
                                "mean_chamfer_distance"
                            ],
                            "eval/std_chamfer_distance": metrics_dict[
                                "std_chamfer_distance"
                            ],
                        }
                    )
            except Exception as e:
                logging.warning(f"Failed to compute metrics at epoch {epoch}: {e}")


if __name__ == "__main__":
    velocity_test()
