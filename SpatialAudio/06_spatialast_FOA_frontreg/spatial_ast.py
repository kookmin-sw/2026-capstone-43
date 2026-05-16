from model import FOASpatialASTModel, build_model


class SpatialAST(FOASpatialASTModel):
    """
    Backward-compatible wrapper.

    The new modular path is:
    - `backbone.py` for FOA feature stem + transformer
    - `heads.py` for task heads
    - `model.py` for composition

    This wrapper keeps the previous tuple-style forward output so existing
    smoke tests and quick experiments still run unchanged.
    """

    def forward(self, waveforms, reverbs=None, mask_t_prob=0.0, mask_f_prob=0.0):
        outputs = super().forward(
            waveforms,
            reverbs=reverbs,
            mask_t_prob=mask_t_prob,
            mask_f_prob=mask_f_prob,
        )
        return (
            outputs.get("class_logits"),
            outputs.get("distance_logits"),
            outputs.get("azimuth_logits", outputs.get("azimuth_regression")),
            outputs.get("elevation_logits"),
            outputs.get("vector"),
        )


def build_AST(**kwargs):
    return SpatialAST(**kwargs)


def build_modular_model(**kwargs):
    return build_model(**kwargs)
