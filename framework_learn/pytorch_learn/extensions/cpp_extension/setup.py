import os
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from Vision import VISION_ROOT

if platform.system() == "Windows":
    os.environ["DISTUTILS_USE_SDK"] = "1"

setup(
    name="cpp_extension_learn",
    ext_modules=[
        CppExtension(
            "cpp_extension_learn",
            [str(VISION_ROOT / "framework_learn/pytorch_learn/extensions/cpp_extension/src/cpp_extension_learn.cpp")],
            include_dirs=[str(VISION_ROOT / "framework_learn/pytorch_learn/extensions/cpp_extension/include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
