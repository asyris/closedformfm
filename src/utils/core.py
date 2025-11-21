
import functools

def hydra_main(config_name, no_runname=False, config_path="../conf", version_base=None):

    def decorator(func):
        import hydra
        @hydra.main(
            config_name=config_name,
            config_path=config_path,
            version_base=version_base
        )
        @functools.wraps(func)
        def wrapper(cfg, *args, **kwargs):
            if not no_runname:
                print(f"## run_name: {cfg.run_name}")
            return func(cfg, *args, **kwargs)
        return wrapper
    return decorator

