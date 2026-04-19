"""ForgeRL — Setup configuration for pip installation."""

from setuptools import setup, find_packages

setup(
    name="forgerl",
    version="1.0.0",
    description="Multi-Agent Software Engineering RL Environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ForgeRL Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "google-generativeai>=0.8.0",
        "pytest>=8.0.0",
        "httpx>=0.27.0",
        "matplotlib>=3.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "training": [
            "unsloth",
            "trl>=0.12.0",
            "datasets>=2.18.0",
            "torch>=2.0.0",
        ],
        "deploy": [
            "gradio>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forgerl-demo=demo.run_demo:main",
            "forgerl-server=forge_env.server:main",
        ],
    },
)
