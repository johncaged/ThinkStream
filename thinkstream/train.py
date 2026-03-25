import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from slyme.context import Context
from deepslyme.utils.config.argparse import parse_and_inject
import thinkstream.model.patch  # noqa
from thinkstream.model.streaming_attention import register_streaming_attention

register_streaming_attention()
# Import the registry containing our builders
from thinkstream.trainer.builder import TRAINER_BUILDERS


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: torchrun ... train.py <builder_name> [args...]")
        print(f"Available builders: {list(TRAINER_BUILDERS.keys())}")
        sys.exit(1)

    builder_name = sys.argv[1]

    try:
        # Retrieve the builder from the registry
        builder_fn = TRAINER_BUILDERS.get(builder_name)
    except KeyError:
        print(f"Error: Unknown builder '{builder_name}'.")
        print(f"Available builders: {list(TRAINER_BUILDERS.keys())}")
        sys.exit(1)

    # Build the train node
    train_node = builder_fn()
    ctx = Context()
    # Parse CLI arguments excluding the builder name and inject them into the context
    ctx = parse_and_inject(
        node=train_node,
        extra_refs=[],
        context=ctx,
        cli_args=sys.argv[2:],
    )
    print(train_node)
    # Prepare and execute the training pipeline
    train_node.prepare()(ctx)
