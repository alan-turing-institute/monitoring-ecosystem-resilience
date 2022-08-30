from setuptools import setup

with open("pyveg/requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="pyveg",
    version="1.1.0",
    description="Vegetation patterns study.",
    url="https://github.com/alan-turing-institute/monitoring-ecosystem-resilience",
    author="Nick Barlow, Camila Rangel Smith and Samuel Van Stroud",
    license="MIT",
    include_package_data=True,
    packages=["pyveg", "pyveg.src", "pyveg.scripts", "pyveg.configs"],
    install_requires=REQUIRED_PACKAGES,
    scripts=["pyveg/scripts/batch_commands.sh"],
    entry_points={
        "console_scripts": [
            "pyveg_run_pipeline=pyveg.scripts.run_pyveg_pipeline:main",
            "pyveg_run_module=pyveg.scripts.run_pyveg_module:main",
            "pyveg_generate_config=pyveg.scripts.generate_config_file:main",
            "pyveg_run_pipeline_loop=pyveg.scripts.run_pipeline_loop:main",
        ]
    },
)
