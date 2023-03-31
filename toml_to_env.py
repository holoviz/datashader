try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


env_name = "bk-test"

channels = [
    "pyviz/label/dev",
    "conda-forge",
    "nodefaults",
]

# Mapping from pip to conda requirements. Previously in setup.cfg
mapping = dict(dask="dask-core")


def toml_to_env():
    """
    Convert pyproject.toml dependencies into a conda environment yaml file.
    """
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject["project"]["dependencies"]
    options = pyproject["project"]["optional-dependencies"]
    required_deps = sorted(deps + options["tests"])

    # Need to check for deps like "datashader[examples]"

    with open("env.yaml", "w") as f:
        f.write(f"name: {env_name}\n")
        if channels:
            f.write("channels:\n")
            for channel in channels:
                f.write(f"  - {channel}\n")
        f.write("dependencies:\n")
        for dep in required_deps:
            dep = mapping.get(dep, dep)
            f.write(f"  - {dep}\n")


if __name__ == '__main__':
    toml_to_env()
