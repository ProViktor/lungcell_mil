import os
import hydra
from omegaconf import DictConfig, OmegaConf
from optimize_hyper import OptimizeHyper
from mil.schemas import HyperRunParams


@hydra.main(version_base=None, config_path="../configs", config_name="hyper_optim")
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to plain dict for Pydantic validation
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Merge common_settings and optimizer_settings into the top-level dict for HyperRunParams
    params_data = {
        **cfg_dict["common_settings"],
        **cfg_dict["optimizer_settings"],
        "aggregator": cfg_dict["aggregator"],
        "pbounds": cfg_dict["pbounds"],
    }

    # Validate with Pydantic
    params = HyperRunParams(**params_data)

    # Initialize and run
    parameter_search = OptimizeHyper()
    parameter_search.run_search(params)


if __name__ == "__main__":
    main()
