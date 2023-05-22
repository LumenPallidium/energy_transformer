import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "energy_transformer",
    version = "0.0.1",
    author = "Keir Havel",
    author_email = "keirhavel@live.com",
    description = "Pytorch implementation of an energy transformer.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/LumenPallidium/energy_transformer",
    project_urls = {"Issue Tracker" : "https://github.com/LumenPallidium/energy_transformer/issues",
                    "Technical Paper" : "https://openreview.net/pdf?id=4nrZXPFN1c4",
                    "Jax Implementation" : "https://github.com/bhoov/energy-transformer-jax"
                    },
    license = "MIT",
    packages = ["energy_transformer"],
    install_requires = ["torch>=2.0"],
)