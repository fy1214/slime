"""Pytest CLI options shared by slime tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--rollout-batch-size",
        action="store",
        type=int,
        default=None,
        help=(
            "Qwen3 e2e rollout_batch_size and global_batch_size. "
            "Default 8, or SLIME_E2E_ROLLOUT_BATCH_SIZE if set."
        ),
    )
