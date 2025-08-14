import ray
import torch

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    assert not args.offload and not args.colocate, "Offload and colocate are not supported for full async RL training."

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    # Note that we initialize it earlier as megatron ckpt loading may have really large peak memory usage.
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    prof = None
    if (
        args.use_pytorch_profiler
        and torch.distributed.get_rank() == 0
    ):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    # async train loop.
    rollout_data_next_future = rollout_manager.async_generate(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.use_pytorch_profiler and torch.distributed.get_rank() == 0:
            prof.step()

        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.async_generate(rollout_id + 1)

        ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        # Profiling.
        if (
            args.use_pytorch_profiler
            and rollout_id == args.profile_step_end
            and torch.distributed.get_rank() == 0
        ):
            assert prof is not None
            prof.stop()

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.data_buffer.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            rollout_data_next_future = None
            ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.async_eval(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
