class ProductionConfig:
    model_name: str
    model_init_kwargs: dict
    resize_hw: tuple
    normalization_mean: tuple
    normalization_std: tuple
    state_dict_path: str
    confidence_threshold: float


def load_finetuned_model(run_id):
    pass
