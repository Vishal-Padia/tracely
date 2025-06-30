from setuptools import setup, find_packages

setup(
    name="tracely",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "pandas",
        "pyarrow",
        "streamlit",
        "plotly",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "tracely = tracely.cli:main",
        ],
    },
)
