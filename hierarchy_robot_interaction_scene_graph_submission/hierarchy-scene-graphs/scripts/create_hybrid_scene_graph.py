import sys
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hybrid_scene_graph_builder import HybridSceneGraphBuilder
from utils.rgbd_data import resolve_hov_scene_path, resolve_replica_scene_path


STEP_THROUGH_FULL_PCD_FLAG = "--step-through-full-pcd"
HOV_FLAG = "--hov"
REUSE_EXISTING_LAYOUT_FLAG = "--reuse-existing-layout"
DEFAULT_HOV_DATASET_ROOT = PROJECT_ROOT.parent / "my_local_data" / "hssd-HOV"


def extract_runtime_args(argv):
    runtime_options = {
        "step_through_full_pcd": False,
        "use_hov": False,
        "reuse_existing_layout": False,
    }
    remaining_argv = []
    for arg in argv:
        if arg == STEP_THROUGH_FULL_PCD_FLAG:
            runtime_options["step_through_full_pcd"] = True
            continue
        if arg == HOV_FLAG:
            runtime_options["use_hov"] = True
            continue
        if arg == REUSE_EXISTING_LAYOUT_FLAG:
            runtime_options["reuse_existing_layout"] = True
            continue
        remaining_argv.append(arg)
    return remaining_argv, runtime_options


def load_config(argv):
    argv, runtime_options = extract_runtime_args(argv)
    default_config_path = PROJECT_ROOT / "config" / "create_hybrid_scene_graph.yaml"
    config_path = default_config_path
    overrides = argv
    if argv and argv[0].endswith((".yaml", ".yml")):
        config_path = Path(argv[0])
        overrides = argv[1:]

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.pipeline.step_through_full_pcd = bool(
        runtime_options["step_through_full_pcd"]
        or cfg.pipeline.get("step_through_full_pcd", False)
    )
    cfg.pipeline.reuse_existing_layout = bool(
        runtime_options["reuse_existing_layout"]
        or cfg.pipeline.get("reuse_existing_layout", False)
    )
    if runtime_options["use_hov"]:
        cfg.main.dataset = "replica_hov"
        cfg.main.dataset_path = str(DEFAULT_HOV_DATASET_ROOT)
    return cfg


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config(argv)
    cfg.main.scene_id = str(cfg.main.scene_id)

    if cfg.main.dataset == "replica":
        scene_path = resolve_replica_scene_path(cfg.main.dataset_path, cfg.main.scene_id)
    elif cfg.main.dataset == "replica_hov":
        scene_path = resolve_hov_scene_path(cfg.main.dataset_path, cfg.main.scene_id)
    else:
        raise ValueError(
            "This script currently supports 'replica' and 'replica_hov' RGB-D inputs only."
        )
    cfg.main.dataset_path = str(scene_path)
    cfg.main.scene_id = scene_path.name

    builder = HybridSceneGraphBuilder(cfg)
    save_dir = builder.run()
    print(f"saved hybrid graph to {save_dir}")


if __name__ == "__main__":
    main()
