"""Stage-2 LeWM training: proposal + refinement planner with frozen world model."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from app.vjepa_cowa_world_model.models import RefinementDecoder
from app.vjepa_cowa_world_model.models.proposal_providers import build_proposal_provider
from app.vjepa_cowa_world_model.training import (
    OPEN_LOOP_SELECTION_RULE,
    TrainingTimer,
    add_planner_param_groups,
    build_validation_record,
    calculate_iterations_per_epoch,
    compile_models,
    create_optimizer_and_scheduler,
    create_stage_timing_meters,
    create_train_dataloader,
    create_transforms,
    create_val_dataloader,
    create_validation_transforms,
    freeze_parameters,
    get_encoder_embed_dim,
    get_next_batch,
    init_encoder,
    init_proposal_encoder,
    is_better_open_loop_candidate,
    is_epoch_validation_due,
    load_checkpoint,
    load_clips,
    load_pretrained_checkpoint,
    log_stage_timing_summary,
    log_trainable_parameters,
    log_training_summary,
    maybe_run_gc,
    parse_training_config,
    record_stage_timing,
    resolve_timing_warmup_iters,
    resume_from_checkpoint,
    start_cuda_timing,
    stop_cuda_timing,
    wait_for_checkpoint_save,
    wrap_ddp_models,
)
from app.vjepa_cowa_world_model.training.config import (
    is_vjepa_main_encoder_config,
    resolve_proposal_num_time_steps,
    resolve_proposal_runtime_normalize_reps,
    resolve_proposal_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.models import (
    configure_vjepa_encoder_trainability,
    init_predictor_runtime_with_token_ae,
    resolve_main_predictor_runtime_overrides,
)
from app.vjepa_cowa_world_model.training.pipeline import setup_run
from app.vjepa_cowa_world_model.training.predictor_parallel import maybe_register_parallel_predictor_tokens
from app.vjepa_cowa_world_model.training.runtimes.loop_runner import (  # noqa: E402
    TrainingLoopRunner,
    resolve_periodic_checkpoint_epoch,
)
from app.vjepa_cowa_world_model.training.runtimes.refinement_runtime import (
    _planner_command_dim,
    build_gt_trajectory,
    build_proposal_history,
    build_stage_predictor_rollout_fn,
    build_status_feature,
    call_planner_method,
    compute_multimodal_proposal_loss,
    forward_frozen_proposal,
    freeze_module_eval,
    gather_mode_trajectory,
    load_frozen_proposal_encoder,
    load_frozen_proposal_provider,
    load_proposal_context_clips,
    pairwise_diversity_hinge_loss,
    requires_proposal_context_clips,
    resolve_proposal_token_ae_module,
    run_stage2_refinement_for_training,
    save_stage_checkpoint,
    select_observed_context_clips,
    unwrap_module,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    forward_main_context,
    forward_main_context_dual,
    resolve_main_timeline,
)
from app.vjepa_cowa_world_model.training.validation_capabilities import validate_validation_suite_execution_contract
from app.vjepa_cowa_world_model.utils import resolve_effective_planner_status_dim
from app.vjepa_cowa_world_model.val_lewm_staged import (
    log_stage_training_metrics,
    log_stage_validation_metrics,
    run_stage_validation,
)
from src.utils.logging import AverageMeter, get_logger

log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _init_refinement_planner(
    config,
    encoder_dim: int,
    tokens_per_frame: int,
    device: torch.device,
    proposal_encoder_dim: int | None = None,
    proposal_encoder: torch.nn.Module | None = None,
    main_num_context_frames: int | None = None,
):
    num_poses = config.data.num_target_frames - config.train.num_observed_frames
    status_dim = resolve_effective_planner_status_dim(config)
    command_dim = _planner_command_dim(config)
    proposal_tokens_per_frame = resolve_proposal_tokens_per_frame(config, proposal_encoder)
    proposal_num_context_frames = resolve_proposal_num_time_steps(config, proposal_encoder)
    proposal_planner = build_proposal_provider(
        config=config,
        encoder_dim=proposal_encoder_dim or encoder_dim,
        tokens_per_frame=proposal_tokens_per_frame,
        num_poses=num_poses,
        status_dim=status_dim,
        command_dim=command_dim,
        num_context_frames=proposal_num_context_frames,
        num_observed_frames=config.train.num_observed_frames,
    ).to(device)
    freeze_module_eval(proposal_planner)

    planner = RefinementDecoder(
        encoder_dim=encoder_dim,
        tf_d_model=config.planner.tf_d_model,
        tf_d_ffn=config.planner.tf_d_ffn,
        tf_num_layers=config.planner.tf_num_layers,
        tf_num_head=config.planner.tf_num_head,
        tf_dropout=config.planner.tf_dropout,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=num_poses,
        num_context_frames=main_num_context_frames or config.train.num_observed_frames,
        status_dim=status_dim,
        use_spatial_tokens=config.planner.use_spatial_tokens,
        num_modes=config.refinement.num_modes,
        use_temporal=True,
        use_time_aligned_bias=config.planner.temporal_alignment,
        use_status_for_planner=config.planner.use_status_for_planner,
        command_dim=command_dim,
        max_rounds=max(2, config.refinement.inference_num_rounds, config.refinement_gated.num_rounds),
    ).to(device)
    return proposal_planner, planner, num_poses


def main(args, resume_preempt=False):
    del resume_preempt
    config = parse_training_config(args)
    if not config.proposal.enabled:
        raise ValueError("Stage-2 refinement requires proposal.enabled=true")
    if not config.proposal.freeze:
        raise ValueError("Stage-2 refinement expects proposal.freeze=true")
    if config.proposal.provider_type != "history_kinematic" and not config.proposal.checkpoint:
        raise ValueError("Stage-2 frozen transformer/diffusion proposal provider requires proposal.checkpoint")
    config.planner.use_planner = True
    config.planner.use_z_context = True
    config.refinement.num_modes = config.proposal.num_modes
    config.planner.num_modes = config.proposal.num_modes
    config.meta.load_planner = False
    config.segmentation.use_segmentation = False

    validate_validation_suite_execution_contract(
        config,
        line_name="refinement",
        declared_executors=set(),
        active_consumers={"planner"},
    )

    ctx = setup_run(config)
    world_size, rank, device, ckpt_paths = ctx.world_size, ctx.rank, ctx.device, ctx.ckpt_paths
    csv_logger, tb_writer = ctx.csv_logger, ctx.tb_writer
    latest_path = ckpt_paths["latest"]
    best_path = os.path.join(config.meta.folder, "best_open_loop.pt")

    best_ade = best_fde = best_minade_k = best_minfde_k = float("inf")
    best_l2_avg = float("inf")
    best_collision_rate = float("inf")
    best_epoch = 0
    best_open_loop_record = None
    validation_history = []

    encoder, target_encoder = init_encoder(config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(config, encoder)
    main_context_timeline = resolve_main_timeline(
        config, encoder=encoder, num_raw_frames=config.train.num_observed_frames
    )
    logger.info(
        "Main encoder timeline: raw_observed=%d stride=%d observed_steps=%d tokens_per_step=%d",
        main_context_timeline.raw_num_frames,
        main_context_timeline.frame_stride,
        main_context_timeline.num_observed_steps,
        main_context_timeline.tokens_per_frame,
    )
    proposal_encoder = None
    proposal_encoder_embed_dim = encoder_embed_dim
    if config.proposal.use_separate_encoder and config.proposal.provider_type != "history_kinematic":
        proposal_encoder = init_proposal_encoder(config, device)
        proposal_encoder_embed_dim = get_encoder_embed_dim(proposal_encoder)
        freeze_module_eval(proposal_encoder)
    elif config.proposal.use_separate_encoder:
        logger.info("Skipping separate proposal_encoder for history_kinematic proposal provider.")
    predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
        config,
        device=device,
        encoder_embed_dim=encoder_embed_dim,
        raw_tokens_per_frame_override=main_tokens_override,
        predictor_img_size_override=predictor_img_size_override,
    )
    raw_future_frames = config.data.num_target_frames - config.train.num_observed_frames
    if raw_future_frames % main_context_timeline.frame_stride != 0:
        raise ValueError(
            "Stage-2 parallel predictor future horizon must be divisible by main encoder frame stride: "
            f"future_frames={raw_future_frames}, frame_stride={main_context_timeline.frame_stride}"
        )
    maybe_register_parallel_predictor_tokens(
        predictor=predictor,
        config=config,
        embed_dim=encoder_embed_dim,
        future_steps=raw_future_frames // main_context_timeline.frame_stride,
        tokens_per_frame=tokens_per_frame,
        device=device,
    )
    proposal_runtime_normalize_reps = resolve_proposal_runtime_normalize_reps(config)
    proposal_token_ae = resolve_proposal_token_ae_module(config, token_ae)
    proposal_planner, planner, num_poses = _init_refinement_planner(
        config,
        encoder_embed_dim,
        tokens_per_frame,
        device,
        proposal_encoder_embed_dim,
        proposal_encoder,
        main_num_context_frames=main_context_timeline.num_observed_steps,
    )
    configure_vjepa_encoder_trainability(encoder, config, trainable=False)
    configure_vjepa_encoder_trainability(target_encoder, config, trainable=False)

    compile_models(encoder, target_encoder, predictor, None, config.model.compile_model)
    if proposal_encoder is not None and config.model.compile_model:
        proposal_encoder.compile()
    transform = create_transforms(config)
    validation_transform = create_validation_transforms(config)
    train_loader, train_sampler = create_train_dataloader(config, rank, world_size, transform)
    val_loader, val_sampler = create_val_dataloader(config, rank, world_size, validation_transform)
    ipe = calculate_iterations_per_epoch(config, train_loader)

    optimizer, scaler, scheduler, wd_scheduler = create_optimizer_and_scheduler(
        config,
        encoder,
        predictor,
        None,
        None,
        planner,
        ipe,
    )
    add_planner_param_groups(optimizer, planner)

    models = wrap_ddp_models(
        encoder,
        target_encoder,
        predictor,
        None,
        None,
        planner,
        encoder_train=False,
        use_planner=True,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_temporal=True,
        use_z_context=True,
    )
    encoder = models["encoder"]
    target_encoder = models["target_encoder"]
    predictor = models["predictor"]
    planner = models["planner"]
    configure_vjepa_encoder_trainability(encoder, config, trainable=False)
    configure_vjepa_encoder_trainability(target_encoder, config, trainable=False)

    freeze_parameters(
        encoder,
        target_encoder,
        predictor,
        None,
        None,
        planner,
        encoder_train=False,
        predictor_train=config.refinement.predictor_finetune,
        seg_head_train=False,
    )
    log_trainable_parameters(
        encoder,
        predictor,
        None,
        None,
        planner,
        optimizer,
        use_planner=True,
        predictor_state_mode="staged_refinement",
    )

    load_pretrained_checkpoint(
        config.meta.pretrain_checkpoint_full,
        encoder,
        target_encoder,
        predictor,
        None,
        None,
        planner,
        load_encoder=config.meta.load_encoder and not is_vjepa_main_encoder_config(config),
        load_predictor=config.meta.load_predictor,
        load_seg=False,
        load_planner=False,
        context_encoder_key=config.meta.context_encoder_key,
        target_encoder_key=config.meta.target_encoder_key,
        rank=rank,
        world_size=world_size,
        predictor_checkpoint=config.meta.predictor_checkpoint,
    )
    load_frozen_proposal_encoder(proposal_encoder, config)
    load_frozen_proposal_provider(proposal_planner, config.proposal.checkpoint, config=config)
    start_epoch = 0
    if config.meta.resume_checkpoint:
        resume_path = ckpt_paths["resume"]
        if resume_path is None:
            logger.warning("Resume checkpoint not found: %s", config.meta.resume_checkpoint)
        else:
            encoder_module = unwrap_module(encoder)
            target_module = unwrap_module(target_encoder)
            predictor_module = unwrap_module(predictor)
            planner_module_for_resume = unwrap_module(planner)
            start_epoch = resume_from_checkpoint(
                resume_path,
                encoder_module,
                target_module,
                predictor_module,
                None,
                None,
                planner_module_for_resume,
                optimizer,
                scaler,
                scheduler,
                wd_scheduler,
                use_planner=True,
                load_planner=(not config.meta.resume_model_only) or config.meta.load_planner,
                rank=rank,
                world_size=world_size,
                model_only=config.meta.resume_model_only,
            )
            target_module.load_state_dict(encoder_module.state_dict())
            logger.info("Synchronized target_encoder with encoder after staged resume")
            # Restore best-metric tracker from best_path so a resumed run does not
            # overwrite a real best checkpoint on its first validation. best_path
            # embeds best_open_loop_record in extra_state; resumed latest.pt does not.
            if not config.meta.resume_model_only:
                _best_ol_ckpt = load_checkpoint(best_path)
                if _best_ol_ckpt is not None and _best_ol_ckpt.get("best_open_loop_record") is not None:
                    best_open_loop_record = _best_ol_ckpt["best_open_loop_record"]
                    best_ade = best_open_loop_record.get("ade", float("inf"))
                    best_fde = best_open_loop_record.get("fde", float("inf"))
                    best_minade_k = best_open_loop_record.get("minade_k", float("inf"))
                    best_minfde_k = best_open_loop_record.get("minfde_k", float("inf"))
                    best_l2_avg = best_open_loop_record.get("l2_avg", float("inf"))
                    best_collision_rate = best_open_loop_record.get("collision_rate", float("inf"))
                    best_epoch = best_open_loop_record.get("epoch", 0)
                    logger.info(
                        "Restored best open-loop tracker from %s (epoch=%s, L2_avg=%.5f)",
                        best_path,
                        best_epoch,
                        best_l2_avg,
                    )
    freeze_module_eval(proposal_planner)
    if proposal_encoder is not None:
        freeze_module_eval(proposal_encoder)

    planner_module = unwrap_module(planner)

    timer = TrainingTimer()
    timing_warmup_iters = resolve_timing_warmup_iters("LEWM_TIMING_WARMUP_ITERS")
    loader = iter(train_loader)
    # Phase 5: main-scope so the checkpoint callbacks + runner gating see the latest epoch's values.
    meters = None
    has_validation = False

    def _stage_barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def run_epoch(epoch):
        nonlocal loader, meters, has_validation
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        meters = {
            "loss": AverageMeter(),
            "prop": AverageMeter(),
            "refine": AverageMeter(),
            "anchor": AverageMeter(),
            "div": AverageMeter(),
        }
        timing_meters = create_stage_timing_meters()

        for itr in range(ipe):
            timer.start_iteration()
            loader, sample, _ = get_next_batch(loader, train_loader, train_sampler, epoch)
            data_elapsed_time_ms = timer.record_data_load()
            gpu_start_event, gpu_end_event = start_cuda_timing(enabled=(itr % log_freq == 0 or itr == ipe - 1))
            maybe_run_gc(itr, GARBAGE_COLLECT_ITR_FREQ, config.meta.sync_gc)
            context_clips, actions, states, extrinsics, _, driving_command, ego_dynamics = load_clips(
                sample,
                device=device,
                use_segmentation=False,
                dtype=torch.float,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            dt = 1.0 / float(max(config.data.fps, 1))
            observed_context_clips = select_observed_context_clips(context_clips, config.train.num_observed_frames)
            proposal_context_clips = load_proposal_context_clips(
                sample,
                context_clips,
                device=device,
                dtype=torch.float,
                require_proposal_context=requires_proposal_context_clips(proposal_encoder),
            )
            observed_proposal_context_clips = select_observed_context_clips(
                proposal_context_clips,
                config.train.num_observed_frames,
            )
            use_shared_proposal_context = (
                proposal_encoder is None and config.proposal.provider_type != "history_kinematic"
            )

            with torch.cuda.amp.autocast(dtype=config.dtype, enabled=config.mixed_precision):
                if use_shared_proposal_context:
                    z_context, z_context_proposal = forward_main_context_dual(
                        encoder,
                        observed_context_clips,
                        config=config,
                        predictor_normalize_reps=runtime_normalize_reps,
                        proposal_normalize_reps=proposal_runtime_normalize_reps,
                        predictor_token_ae=token_ae,
                        proposal_token_ae=proposal_token_ae,
                    )
                else:
                    z_context = forward_main_context(
                        encoder,
                        observed_context_clips,
                        config=config,
                        runtime_normalize_reps=runtime_normalize_reps,
                        token_ae=token_ae,
                    )
                    z_context_proposal = None
                status_feature = build_status_feature(config, states, actions, driving_command, ego_dynamics)
                gt_trajectory = build_gt_trajectory(config, states, num_poses)
                history_traj = build_proposal_history(config, actions, dt)

                if use_shared_proposal_context:
                    with torch.no_grad():
                        proposal_out = proposal_planner(
                            z_context=z_context_proposal,
                            status_feature=status_feature,
                            history_traj=history_traj,
                        )
                else:
                    proposal_out = forward_frozen_proposal(
                        proposal_encoder=proposal_encoder,
                        proposal_planner=proposal_planner,
                        context_clips=observed_proposal_context_clips,
                        use_tubelet_repeat=config.data.use_tubelet_repeat,
                        proposal_normalize_reps=proposal_runtime_normalize_reps,
                        proposal_token_ae=proposal_token_ae,
                        status_feature=status_feature,
                        history_traj=history_traj,
                        num_observed_frames=config.train.num_observed_frames,
                    )

                from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output

                # Fail-loud proposal contract before indexing (verified: providers return trajectories
                # [B, K, num_poses, 3] + confidences [B, K]; maybe_expand_manual_proposal preserves it).
                validate_planner_output(proposal_out, mode="inference")
                traj_m = proposal_out["trajectories"]
                logit_m = proposal_out["confidences"]
                out1 = proposal_out.get("proposal_features")
                if out1 is None or out1.shape[-1] != planner_module.tf_d_model:
                    out1 = planner_module.encode_proposal_features(traj_m)

                _predictor_rollout_fn = build_stage_predictor_rollout_fn(
                    stage="stage2",
                    predictor=predictor,
                    z_context=z_context,
                    actions=actions,
                    states=states,
                    driving_command=driving_command,
                    ego_dynamics=ego_dynamics,
                    config=config,
                    tokens_per_frame=tokens_per_frame,
                    runtime_normalize_reps=runtime_normalize_reps,
                    dt=dt,
                    predictor_observed_steps=main_context_timeline.num_observed_steps,
                    predictor_frame_stride=main_context_timeline.frame_stride,
                )

                traj_input = traj_m.detach() if config.refinement.refine_stop_grad_to_proposer else traj_m
                logit_input = logit_m.detach() if config.refinement.refine_stop_grad_to_proposer else logit_m
                out1_input = out1.detach() if config.refinement.refine_stop_grad_to_proposer else out1
                traj_final = run_stage2_refinement_for_training(
                    planner=planner,
                    config=config,
                    z_context=z_context,
                    status_feature=status_feature,
                    proposal_trajs=traj_input,
                    proposal_logits=logit_input,
                    proposal_features=out1_input,
                    predictor_rollout_fn=_predictor_rollout_fn,
                    call_planner_method_fn=call_planner_method,
                )

                prop_result = compute_multimodal_proposal_loss(config, traj_m, logit_m, gt_trajectory, epoch)
                refine_loss = F.smooth_l1_loss(traj_final, gt_trajectory)
                winner_traj = gather_mode_trajectory(traj_m, prop_result["winner_idx"])
                anchor_loss = F.smooth_l1_loss(traj_final, winner_traj.detach())
                div_loss = pairwise_diversity_hinge_loss(traj_m)

                loss = config.refinement.lambdas.refine * refine_loss + config.refinement.lambdas.anchor * anchor_loss

            optimizer.zero_grad(set_to_none=True)
            if config.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=config.optimization.grad_clip_norm)
            if config.refinement.predictor_finetune:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=config.optimization.grad_clip_norm)
            if config.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if wd_scheduler is not None:
                wd_scheduler.step()

            gpu_elapsed_ms = stop_cuda_timing(gpu_start_event, gpu_end_event)
            iter_elapsed_time_ms, data_elapsed_time_ms = timer.stop_iteration()
            train_elapsed_time_ms = max(0.0, iter_elapsed_time_ms - data_elapsed_time_ms)
            record_stage_timing(
                timing_meters,
                itr,
                timing_warmup_iters,
                iter_elapsed_time_ms,
                train_elapsed_time_ms,
                gpu_elapsed_ms,
                data_elapsed_time_ms,
            )
            meters["loss"].update(float(loss.detach()))
            meters["prop"].update(float(prop_result["loss"].detach()))
            meters["refine"].update(float(refine_loss.detach()))
            meters["anchor"].update(float(anchor_loss.detach()))
            meters["div"].update(float(div_loss.detach()))

            if tb_writer is not None and rank == 0:
                global_step = epoch * ipe + itr
                tb_writer.add_scalar("stage2/loss", meters["loss"].val, global_step)
                tb_writer.add_scalar("stage2/prop_loss", meters["prop"].val, global_step)
                tb_writer.add_scalar("stage2/refine_loss", meters["refine"].val, global_step)

            if rank == 0 and (itr % log_freq == 0 or itr == ipe - 1):
                logger.info(
                    "[stage2][epoch %d][iter %d/%d] loss=%.4f prop=%.4f refine=%.4f anchor=%.4f div=%.4f lr=%.2e",
                    epoch + 1,
                    itr + 1,
                    ipe,
                    meters["loss"].avg,
                    meters["prop"].avg,
                    meters["refine"].avg,
                    meters["anchor"].avg,
                    meters["div"].avg,
                    current_lr,
                )

        has_validation = config.planner.use_planner and is_epoch_validation_due(
            val_loader, epoch, config.meta.val_freq
        )
        if rank == 0 and csv_logger is not None:
            log_stage_training_metrics(
                csv_logger,
                epoch,
                {
                    "loss": meters["loss"].avg,
                    "prop": meters["prop"].avg,
                    "refine": meters["refine"].avg,
                    "anchor": meters["anchor"].avg,
                    "div": meters["div"].avg,
                },
                has_validation=has_validation,
            )
        if rank == 0:
            log_stage_timing_summary("stage2", logger, epoch, timing_meters, timing_warmup_iters)

        return has_validation

    def _save_latest(epoch):
        save_stage_checkpoint(
            latest_path,
            encoder,
            target_encoder,
            predictor,
            planner,
            optimizer,
            scaler,
            scheduler,
            wd_scheduler,
            epoch + 1,
            meters["loss"].avg,
            config,
            rank,
            world_size,
        )

    def _save_periodic(epoch):
        periodic_epoch = resolve_periodic_checkpoint_epoch(
            epoch,
            periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
            selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        )
        save_stage_checkpoint(
            os.path.join(config.meta.folder, f"e{periodic_epoch}.pt"),
            encoder,
            target_encoder,
            predictor,
            planner,
            optimizer,
            scaler,
            scheduler,
            wd_scheduler,
            epoch + 1,
            meters["loss"].avg,
            config,
            rank,
            world_size,
        )

    def _run_validation_isolated(epoch):
        nonlocal best_open_loop_record, best_ade, best_fde, best_minade_k, best_minfde_k
        nonlocal best_l2_avg, best_collision_rate, best_epoch
        if has_validation:
            logger.info("[stage2] Running validation at epoch %d...", epoch + 1)
            val_metrics = run_stage_validation(
                encoder=encoder,
                predictor=predictor,
                proposal_planner=proposal_planner,
                proposal_encoder=proposal_encoder,
                planner=planner,
                val_loader=val_loader,
                val_sampler=val_sampler,
                config=config,
                epoch=epoch + 1,
                rank=rank,
                world_size=world_size,
                tokens_per_frame=tokens_per_frame,
                runtime_normalize_reps=runtime_normalize_reps,
                token_ae=token_ae,
                proposal_runtime_normalize_reps=proposal_runtime_normalize_reps,
                proposal_token_ae=proposal_token_ae,
                stage="stage2",
                vis_output_dir=os.path.join(config.meta.folder, "val_vis"),
            )
            current_record = build_validation_record(epoch + 1, val_metrics)
            validation_history.append(current_record)
            log_stage_validation_metrics(
                tb_writer=tb_writer,
                csv_logger=csv_logger,
                val_metrics=val_metrics,
                epoch=epoch,
                rank=rank,
                validation_history=validation_history,
            )

            if is_better_open_loop_candidate(current_record, best_open_loop_record):
                best_open_loop_record = current_record
                best_ade = current_record.get("ade", float("inf"))
                best_fde = current_record.get("fde", float("inf"))
                best_minade_k = current_record.get("minade_k", float("inf"))
                best_minfde_k = current_record.get("minfde_k", float("inf"))
                best_l2_avg = current_record.get("l2_avg", float("inf"))
                best_collision_rate = current_record.get("collision_rate", float("inf"))
                best_epoch = current_record["epoch"]
                save_stage_checkpoint(
                    best_path,
                    encoder,
                    target_encoder,
                    predictor,
                    planner,
                    optimizer,
                    scaler,
                    scheduler,
                    wd_scheduler,
                    epoch + 1,
                    meters["loss"].avg,
                    config,
                    rank,
                    world_size,
                    extra_state={
                        "best_open_loop_metrics": val_metrics,
                        "best_open_loop_record": current_record,
                        "best_open_loop_metric_schema": "w4d_l2_primary_point_l2_aux_v1",
                    },
                )
                logger.info(
                    "*** [stage2] New best open-loop checkpoint! Epoch %d | Rule: %s | "
                    "L2_avg: %.5f, Collision: %.5f, ADE: %.5f, FDE: %.5f, minFDE@K: %.5f ***",
                    best_epoch,
                    OPEN_LOOP_SELECTION_RULE,
                    best_l2_avg,
                    best_collision_rate,
                    best_ade,
                    best_fde,
                    best_minfde_k,
                )

    def _run_validation(epoch):
        from app.vjepa_cowa_world_model.utils.eval_determinism import preserve_eval_rng_state

        with preserve_eval_rng_state(device):
            return _run_validation_isolated(epoch)

    TrainingLoopRunner(
        start_epoch=start_epoch,
        epochs=config.optimization.epochs,
        ipe=ipe,
        rank=rank,
        checkpoint_freq=CHECKPOINT_FREQ,
        save_every_freq=config.meta.save_every_freq,
        save_from_epoch=config.meta.save_from_epoch,
        periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
        selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        barrier=_stage_barrier,
    ).run(
        run_epoch=run_epoch,
        save_latest=_save_latest,
        save_periodic=_save_periodic,
        run_validation=_run_validation,
    )

    wait_for_checkpoint_save()
    log_training_summary(
        best_epoch,
        best_ade,
        best_fde,
        best_minade_k,
        best_minfde_k,
        best_path,
        tb_writer,
        rank,
        best_l2_avg=best_l2_avg,
        best_collision_rate=best_collision_rate,
        validation_history=validation_history,
        best_selector_label=OPEN_LOOP_SELECTION_RULE,
    )
