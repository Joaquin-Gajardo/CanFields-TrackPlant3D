import argparse
import json
import logging
import math
import os
import random
import sys
import time

import point_cloud_utils as pcu
import pytorch3d
import torch
import torch.autograd as ag
import trimesh
from pytorch3d.io import IO

from losses.recon import compute_truncated_chamfer_distance
from networks.multi_velocity_debug import adjust_net
from utils import *


def build_args():
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--specifications",
        default="configs/specs.json",
        dest="specifications",
        required=False,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--base_results_dir",
        default="results",
        required=False,
        help="The base results directory",
    )

    arg_parser.add_argument(
        "--methods",
        "-me",
        default="ode_solve",
        dest="methods",
        required=False,
        help="method name (default: ode_solve). Other options: 4DSDF,TPSR,NVFI,4DGF, ode_solve_woAdaptor",
    )
    arg_parser.add_argument(
        "--object_name",
        "-o",
        dest="object_name",
        required=False,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--experiment_name",
        "-en",
        dest="experiment_name",
        required=False,
        help="Name of the specific experiment (e.g., tomato2_control_plant2_leaf1_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_8_stride_1)",
    )
    arg_parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,
        help="please choose 'animals','medical' or 'defaust' ",
    )
    arg_parser.add_argument(
        "--epoch",
        "-e",
        dest="epoch",
        required=True,
        help="Checkpoint selection: 'all' for all checkpoints, 'last' for most recent, or specific epoch number (e.g., 15000)",
    )
    arg_parser.add_argument(
        "--step",
        "-s",
        dest="step",
        required=True,
        help="step_1 canonic shape step:2 move canonic shape",
    )
    arg_parser.add_argument(
        "--multi_step",
        "-m",
        dest="multi_step",
        required=True,
        help="step_1 canonic shape step:2 move canonic shape",
    )
    arg_parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing PLY files (default: False, skip if exists)",
    )

    add_common_args(arg_parser)
    args = arg_parser.parse_args()

    # Use different specs for non-ode_solve methods
    if args.methods != "ode_solve" and not args.methods.startswith("ode_solve_"):
        args.specifications = (args.specifications).replace(
            "specs.json", "others_specs.json"
        )

    configure_logging(args)
    return args


def clear_folder(folder_path):
    contents = os.listdir(folder_path)

    for item in contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            clear_folder(item_path)
            os.rmdir(item_path)
            print(f"Deleted folder: {item_path}")


def get_available_checkpoints(experiment_directory):
    """
    Discover available checkpoint files in the experiment directory.
    Returns a sorted list of epoch numbers.
    """
    checkpoints_dir = os.path.join(
        experiment_directory, "checkpoints", "ModelParameters"
    )
    if not os.path.exists(checkpoints_dir):
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")

    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoints_dir}")

    # Extract epoch numbers from filenames (e.g., 15000.pth -> 15000)
    epochs = []
    for f in checkpoint_files:
        epoch_str = f.replace(".pth", "")
        try:
            epochs.append(int(epoch_str))
        except ValueError:
            continue

    if not epochs:
        raise ValueError(
            f"Could not parse epoch numbers from checkpoint files in {checkpoints_dir}"
        )

    return sorted(epochs)


def select_epochs_to_evaluate(epoch_arg, experiment_directory):
    """
    Based on the epoch argument, return a list of epochs to evaluate.

    Args:
        epoch_arg: String - 'all', 'last', or a specific epoch number
        experiment_directory: Path to the experiment directory

    Returns:
        List of epoch numbers to evaluate
    """
    available_epochs = get_available_checkpoints(experiment_directory)

    if epoch_arg.lower() == "all":
        return available_epochs
    elif epoch_arg.lower() == "last":
        return [available_epochs[-1]]
    else:
        # Specific epoch number
        try:
            specific_epoch = int(epoch_arg)
            if specific_epoch not in available_epochs:
                print(
                    f"Warning: Requested epoch {specific_epoch} not found. Available epochs: {available_epochs}"
                )
                print(f"Using closest available epoch...")
                # Find closest epoch
                closest_epoch = min(
                    available_epochs, key=lambda x: abs(x - specific_epoch)
                )
                return [closest_epoch]
            return [specific_epoch]
        except ValueError:
            raise ValueError(
                f"Invalid epoch argument: {epoch_arg}. Must be 'all', 'last', or a number."
            )


def load_model_latents(args, epoch_num):
    """
    Load model and latents for a specific epoch.

    Args:
        args: Command line arguments
        epoch_num: Specific epoch number to load
    """

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    specs = json.load(open(args.specifications))
    sub_obj_name = args.object_name.split("_full_l2")[0]
    obj_name = args.object_name

    if args.dataset_name == "animals":

        from datasets import data_wmw

        anima_path = "datasets/Deforming4D/notexture/"
        specs["ShapePath"] = anima_path + sub_obj_name
        if args.methods != "ode_solve" or "_woAdaptor" in obj_name:
            gap_lenth = int(obj_name.split("_")[-4])
            start_idx = int(obj_name.split("_")[-6])
            stride = int(obj_name.split("_")[-2])
        else:
            gap_lenth = int(obj_name.split("_")[-3])
            start_idx = int(obj_name.split("_")[-5])
            stride = int(obj_name.split("_")[-1])

        sdf_dataset = data_wmw.DeformSamples(
            specs["ShapePath"], start_idx=start_idx, gap=gap_lenth, stride=stride
        )

    elif args.dataset_name == "defaust":
        from datasets import data_dfaust

        defaust_path = "datasets/D_Faust/"
        specs["ShapePath"] = defaust_path + args.object_name
        sdf_dataset = data_dfaust.DeformSamples(specs["ShapePath"], train_step=3)
        obj_name = (
            specs["ShapePath"].split("/")[-1] + "_S3_full_l2_0.25_elas_20.0_prob_0.12"
        )
    elif args.dataset_name == "medical":
        from datasets import data_medician

        medical_path = "datasets/Medical/"
        specs["ShapePath"] = medical_path + args.object_name
        sdf_dataset = data_medician.DeformSamples(specs["ShapePath"], train_step=1)
        obj_name = (
            specs["ShapePath"].split("/")[-1] + "_M1_full_l2_0.25_elas_10.0_prob_0.18"
        )
    elif args.dataset_name == "pheno4d":
        from datasets import data_pheno4d

        # For Pheno4D, use the experiment_name argument
        if args.experiment_name:
            obj_name = args.experiment_name.split("/")[-1]
        else:
            raise ValueError(
                "For pheno4d dataset, --experiment_name argument is required"
            )

        # Extract parameters from experiment name
        # Expected format: object_P4D_full_l2_X_elas_Y_prob_Z_start_id_A_gap_B_stride_C
        parts = obj_name.split("_")

        # Find parameter values
        start_idx = 0
        gap = 8
        stride = 1
        interp_stride = None

        for i, part in enumerate(parts):
            if part == "start" and i + 2 < len(parts) and parts[i + 1] == "id":
                start_idx = int(parts[i + 2])
            elif part == "gap" and i + 1 < len(parts):
                gap = int(parts[i + 1])
            elif part == "stride" and i + 1 < len(parts):
                # Check if this is interp_stride or regular stride
                if i > 0 and parts[i - 1] == "interp":
                    interp_stride = int(parts[i + 1])
                else:
                    stride = int(parts[i + 1])

        # Use the object_name as the dataset path
        specs["ShapePath"] = args.object_name

        # Auto-detect split file to ensure consistent normalization with training
        # The split file ensures we use the same scale/mean as during training
        sequence_name = args.object_name.split("/")[-1]
        split_file = f"configs/point_splits/{sequence_name}_split.json"
        if not os.path.exists(split_file):
            logging.warning(f"Point split file not found: {split_file}")
            logging.warning("Using all points for normalization (may differ from training)")
            split_file = None

        # In interpolation mode, use TEST points for evaluation
        use_train_points = (interp_stride is None)

        if interp_stride is not None:
            logging.info(f"Interpolation mode detected (interp_stride={interp_stride})")
            logging.info("Evaluating on TEST timesteps with TEST points")

        sdf_dataset = data_pheno4d.DeformSamples(
            specs["ShapePath"],
            start_idx=start_idx,
            gap=gap,
            train_step=stride,
            use_dense=False,
            split_file=split_file,  # Use same split as training for consistent normalization
            use_train_points=use_train_points,  # Use test points in interpolation mode
            interp_stride=interp_stride,
        )
    else:
        raise ValueError(
            f"The variable must be in either animals, medical, defaust, or pheno4d"
        )

    start_idx = 0
    end_idx = len(sdf_dataset) - 1

    ### Set paths ###
    # Extract crop and sequence name from object_name path for pheno4d
    # For pheno4d: Pheno4D/processed/v2/organ_sequences/tomato/{sequence_name}
    # For others: use object_name as-is for backwards compatibility
    if args.dataset_name == "pheno4d":
        path_parts = args.object_name.split("/")
        crop = path_parts[-2]  # Extract crop (e.g., "tomato")
        sequence_name = path_parts[-1]  # Extract sequence name

        # For pheno4d, use experiment_name as the folder name
        if args.experiment_name:
            experiment_folder = args.experiment_name.split("/")[-1]
        else:
            raise ValueError(
                "For pheno4d dataset, --experiment_name argument is required"
            )

        # Build path: results/pheno4d/{method}/{crop}/{sequence_name}/{experiment_name}/
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            args.methods,
            crop,
            sequence_name,
            experiment_folder,
        )
    else:
        # Original structure for other datasets
        object_base_name = sub_obj_name
        experiment_folder = obj_name

        # Build experiment directory path: results/{dataset}/{object}/{method}/{experiment_name}/
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            object_base_name,
            args.methods,
            experiment_folder,
        )

    # Build export path based on method, including epoch subdirectory
    export_path = os.path.join(
        experiment_directory,
        "manifold",
        f"epoch_{epoch_num}",
    )

    latent_size = specs["CodeLength"]
    pts_name = str(epoch_num)

    if args.methods != "ode_solve" and not args.methods.startswith("ode_solve_"):
        from networks.other_volocities import Decoder

        decoder = Decoder(
            args.methods,
            latent_size,
            specs["NetworkSpecs"]["warper_kargs"],
            specs["NetworkSpecs"]["decoder_debug_kargs"],
        )
        decoder = decoder.cuda()
        if args.methods == "OF4D":
            start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
        else:
            start_end_vecs = torch.nn.Embedding(1, 128, max_norm=1.0).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        print("experiment_directory: ", experiment_directory)
        load_model_parameters(experiment_directory, pts_name, decoder)
        load_latent_vectors(experiment_directory, pts_name, start_end_vecs)
        return (
            sdf_dataset,
            decoder,
            start_end_vecs,
            start_idx,
            end_idx,
            export_path,
            experiment_directory,
        )
    elif "woAdaptor" in args.methods:
        specs["NetworkArch"] = "multi_velocity_debug"
        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Warper"])

        decoder = arch.Decoder(
            args.multi_step,
            latent_size,
            specs["NetworkSpecs"]["warper_kargs"],
            specs["NetworkSpecs"]["decoder_debug_kargs"],
        )
        decoder = decoder.cuda()
        start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        load_model_parameters(experiment_directory, pts_name, decoder)
        load_latent_vectors(experiment_directory, pts_name, start_end_vecs)
        return (
            sdf_dataset,
            decoder,
            start_end_vecs,
            start_idx,
            end_idx,
            export_path,
            experiment_directory,
        )
    else:
        specs["NetworkArch"] = "multi_velocity_debug"
        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Warper"])

        decoder = arch.Decoder(
            args.multi_step,
            latent_size,
            specs["NetworkSpecs"]["warper_kargs"],
            specs["NetworkSpecs"]["decoder_debug_kargs"],
        )
        decoder = decoder.cuda()
        # shape_codes inititalize
        ajust = adjust_net().cuda()
        start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
        adjust_start_end_vecs = torch.nn.Embedding(
            end_idx - start_idx + 1, 128, max_norm=1.0
        ).cuda()
        composed_model = torch.nn.ModuleList().cuda()
        composed_model.append(ajust)
        composed_model.append(decoder)
        # print('comnposed_model: ',composed_model)
        print("experiment_directory: ", experiment_directory)
        load_model_parameters(experiment_directory, pts_name, composed_model)
        load_adjust_latent_vectors(
            experiment_directory, pts_name, adjust_start_end_vecs
        )
        ajust = composed_model[0].eval()
        decoder = composed_model[1].eval()
        return (
            sdf_dataset,
            decoder,
            ajust,
            adjust_start_end_vecs,
            start_idx,
            end_idx,
            export_path,
            experiment_directory,
        )


def creat_colored_canonic_mesh(
    decoder, sdf_dataset, start_idx, end_idx, export_path, methods=None
):
    sw_pts = sdf_dataset.evaluate_train(int(start_idx + end_idx // 2))
    sw_bbox_min = sw_pts.min(0, keepdim=True)[0]
    sw_bbox_max = sw_pts.max(0, keepdim=True)[0]
    # compute the swbbox middle point
    swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
    # compute the half length of diagonal bbox
    swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
    min_swbbox = swbbox_middle - 1.7 * swbbox_len
    # inter_latent = None

    # For 4DSDF, we need to pass the template time
    this_time = None
    if methods == "4DSDF":
        start_time = (sdf_dataset[start_idx])["normed_time"].cuda()
        end_time = (sdf_dataset[end_idx])["normed_time"].cuda()
        this_time = 0.5 * (start_time + end_time)

    with torch.no_grad():
        create_mesh_colored_wmw(
            decoder,
            None,
            os.path.join(export_path, "normalized_tempt"),
            N=200,
            sm_offset=min_swbbox[0],
            sm_len=2 * 1.7 * swbbox_len,
            lag_mean=None,
            lag_scale=None,
            this_time=this_time,
        )


def find_closest_number(numbers, target):
    closest_number = min(numbers, key=lambda x: abs(x - target))
    index = numbers.index(closest_number)
    return closest_number, index


def move_canonic_shape(
    methods, sdf_dataset, decoder, start_time, end_time, gt_time, export_path
):
    # backward_time = torch.linspace(start_time,end_time,50).cuda()
    backward_time = sdf_dataset.normalized_times

    # Ensure gradients are turned off and clean any accumulated gradients
    decoder.warper.turoff_grad()
    if hasattr(decoder.warper, 'clean_grads'):
        decoder.warper.clean_grads()

    # In interpolation mode, only evaluate on TEST timesteps
    eval_indices = []
    for idx in range(len(sdf_dataset)):
        if not sdf_dataset.is_training_timestep(idx):
            eval_indices.append(idx)

    # If no test timesteps (normal mode), evaluate on all
    if len(eval_indices) == 0:
        eval_indices = list(range(len(sdf_dataset)))
        logging.info("Normal mode: Evaluating on all timesteps")
    else:
        logging.info(f"Interpolation mode: Evaluating on {len(eval_indices)} TEST timesteps: {eval_indices}")

    if methods == "4DSDF":

        os.makedirs(export_path, mode=0o777, exist_ok=True)
        # clear_folder(export_path)
        d_s = sdf_dataset.scale
        d_m = sdf_dataset.mean
        for this_idx in eval_indices:
            back_time = backward_time[this_idx]
            sub_file = sdf_dataset.sub_files[this_idx]
            print("sub_file: ", sub_file)
            _, index = find_closest_number(sdf_dataset.normalized_times, back_time)
            sw_pts = sdf_dataset.evaluate_train(int(index))
            sw_bbox_min = sw_pts.min(0, keepdim=True)[0]
            sw_bbox_max = sw_pts.max(0, keepdim=True)[0]
            # compute the swbbox middle point
            swbbox_middle = 0.5 * (sw_bbox_min + sw_bbox_max)
            # compute the half length of diagonal bbox
            swbbox_len = torch.abs(sw_bbox_max - swbbox_middle)
            min_swbbox = swbbox_middle - 1.7 * swbbox_len
            # inter_latent = None
            mesh_path = os.path.join(export_path, str(back_time.cpu().item()))
            print("mesh_path: ", mesh_path)
            with torch.no_grad():
                create_mesh_colored_wmw(
                    decoder,
                    None,
                    mesh_path,
                    N=128,
                    sm_offset=min_swbbox[0],
                    sm_len=2 * 1.7 * swbbox_len,
                    lag_mean=d_m,
                    lag_scale=d_s,
                    this_time=back_time.cuda(),
                )
            # Clean accumulated gradients after each timestep
            if hasattr(decoder.warper, 'clean_grads'):
                decoder.warper.clean_grads()
            torch.cuda.empty_cache()
    else:
        templat_full_path = os.path.join(export_path, "normalized_tempt.ply")

        temp_mesh = trimesh.load(templat_full_path)
        temp_v = np.asarray(temp_mesh.vertices, dtype=np.float32)
        temp_f = np.asarray(temp_mesh.faces)
        temp_v = torch.tensor(temp_v).cuda()
        num_chunks = 5
        d_s = sdf_dataset.scale
        d_m = sdf_dataset.mean
        os.makedirs(export_path, mode=0o777, exist_ok=True)
        # clear_folder(export_path)  # Don't clear folder to preserve normalized template

        time.sleep(3)
        for this_idx in eval_indices:
            back_time = backward_time[this_idx]
            sub_file = sdf_dataset.sub_files[this_idx]
            with torch.no_grad():
                chunks = torch.chunk(temp_v, num_chunks, dim=0)
                v_final = []
                for i, chunk in enumerate(chunks):
                    chunk_final = decoder.warper(chunk, gt_time, back_time)
                    v_final.append(chunk_final)
                v_final = torch.cat(v_final, dim=0) / d_s.cuda() + d_m.cuda()
            if methods == "4DSDF":
                mesh_path = os.path.join(export_path, str(back_time.cpu().item()))
            else:
                mesh_path = os.path.join(export_path, sub_file)

            print("mesh_path: ", mesh_path)
            new_mesh = trimesh.Trimesh(
                vertices=v_final.detach().clone().cpu().numpy(), faces=temp_f
            )
            new_mesh.visual.face_colors = temp_mesh.visual.face_colors
            new_mesh.export(mesh_path)
            # Clean accumulated gradients after each timestep
            if hasattr(decoder.warper, 'clean_grads'):
                decoder.warper.clean_grads()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # python eval_sphere.py -o grizzRSS_Attack0  -d animals
    setup_seed(3407)
    args = build_args()

    # First, we need to get the experiment directory to find available epochs
    # We'll do a quick setup to get the experiment directory path
    specs = json.load(open(args.specifications))

    if args.dataset_name == "animals":
        sub_obj_name = args.object_name.split("_full_l2")[0]
        obj_name = args.object_name
    elif args.dataset_name == "defaust":
        obj_name = args.object_name + "_S3_full_l2_0.25_elas_20.0_prob_0.12"
        sub_obj_name = args.object_name
    elif args.dataset_name == "medical":
        obj_name = args.object_name + "_M1_full_l2_0.25_elas_10.0_prob_0.18"
        sub_obj_name = args.object_name
    elif args.dataset_name == "pheno4d":
        if args.experiment_name:
            obj_name = args.experiment_name.split("/")[-1]
            object_base_name = args.object_name.split("/")[-1]
            sub_obj_name = object_base_name
        else:
            raise ValueError(
                "For pheno4d dataset, --experiment_name argument is required"
            )
    else:
        raise ValueError(
            f"The variable must be in either animals, medical, defaust, or pheno4d"
        )

    # Build experiment directory path
    if args.dataset_name == "pheno4d":
        # Extract crop and sequence name from object_name
        path_parts = args.object_name.split("/")
        crop = path_parts[-2]
        sequence_name = path_parts[-1]
        experiment_folder = args.experiment_name.split("/")[-1]

        # Build path: results/pheno4d/{method}/{crop}/{sequence_name}/{experiment_name}/
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            args.methods,
            crop,
            sequence_name,
            experiment_folder,
        )
    else:
        # Original structure for other datasets
        experiment_folder = obj_name
        object_base_name = sub_obj_name

        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            object_base_name,
            args.methods,
            experiment_folder,
        )

    # Get list of epochs to evaluate
    epochs_to_evaluate = select_epochs_to_evaluate(args.epoch, experiment_directory)
    print(f"Evaluating {len(epochs_to_evaluate)} checkpoint(s): {epochs_to_evaluate}")

    # Loop through each epoch
    skipped_count = 0
    failed_count = 0
    for epoch_num in epochs_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Evaluating epoch {epoch_num}")
        print(f"{'='*80}\n")

        try:
            # Check if PLY files already exist in this epoch directory
            epoch_export_path = os.path.join(
                experiment_directory,
                "manifold",
                f"epoch_{epoch_num}",
            )

            # Check for key files based on step
            if args.step == "1":
                key_file = os.path.join(epoch_export_path, "normalized_tempt.ply")
            else:  # step == "2"
                # Skip only a complete export: on TrackPlant3D, one mesh per ground-truth
                # frame, named after it. A partial export is regenerated.
                existing = (
                    set(os.listdir(epoch_export_path))
                    if os.path.exists(epoch_export_path)
                    else set()
                )
                if args.dataset_name == "pheno4d":
                    gt_dir = os.path.join(args.object_name, "sparse_pc")
                    expected = {f for f in os.listdir(gt_dir) if f.endswith(".ply")}
                    # Meshes are named after the frames, or (DSR) by normalized time
                    exported = {f for f in existing if f.endswith(".ply")} - {"normalized_tempt.ply"}
                    complete = bool(expected) and (
                        expected <= existing
                        or (not exported & expected and len(exported) == len(expected))
                    )
                else:
                    complete = any(
                        f.endswith(".ply") and f != "normalized_tempt.ply"
                        for f in existing
                    )
                key_file = epoch_export_path if complete else None

            if key_file and os.path.exists(key_file) and not args.overwrite:
                print(f"PLY files already exist in {epoch_export_path}")
                print(f"Skipping epoch {epoch_num} (use --overwrite to regenerate)")
                skipped_count += 1
                continue

            if args.methods != "ode_solve" and not args.methods.startswith(
                "ode_solve_"
            ):
                (
                    sdf_dataset,
                    decoder,
                    start_end_vecs,
                    start_idx,
                    end_idx,
                    export_path,
                    exp_dir,
                ) = load_model_latents(args, epoch_num)
            else:
                (
                    sdf_dataset,
                    decoder,
                    ajust,
                    adjust_start_end_vecs,
                    start_idx,
                    end_idx,
                    export_path,
                    exp_dir,
                ) = load_model_latents(args, epoch_num)

            if args.step == "1":
                # Ensure the export directory exists before creating the template
                os.makedirs(export_path, mode=0o777, exist_ok=True)
                creat_colored_canonic_mesh(
                    decoder, sdf_dataset, start_idx, end_idx, export_path, args.methods
                )

            start_time = (sdf_dataset[start_idx])["normed_time"].cuda()
            end_time = (sdf_dataset[end_idx])["normed_time"].cuda()

            decoder.warper.split_time(start_time, end_time)
            gt_time = 0.5 * (start_time + end_time)
            if args.step == "2":
                move_canonic_shape(
                    args.methods,
                    sdf_dataset,
                    decoder,
                    start_time,
                    end_time,
                    gt_time,
                    export_path,
                )

            print(f"\nCompleted evaluation for epoch {epoch_num}")
            print(f"Results saved to: {export_path}\n")

        except Exception as e:
            print(f"\nERROR: Failed to process epoch {epoch_num}: {e}")
            logging.error(f"Failed to process epoch {epoch_num}: {e}")
            failed_count += 1
            continue

    print(f"\n{'='*80}")
    created_count = len(epochs_to_evaluate) - skipped_count - failed_count
    print(
        f"All evaluations completed! Created {created_count}, skipped {skipped_count}, failed {failed_count} (total: {len(epochs_to_evaluate)} epoch(s))"
    )
    print(f"{'='*80}")
