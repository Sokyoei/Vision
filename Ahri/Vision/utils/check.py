import importlib.util


def check_package_installed(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        return True
    else:
        return False
