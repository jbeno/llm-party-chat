from setuptools import setup, find_packages

setup(
    name="llm-party-chat",
    version="0.1.0",  # Incremented for significant feature additions
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "websockets>=12.0,<15.0",
        "transformers>=4.37.0,<5.0",
        "torch>=2.1.0,<3.0",
        "colorama>=0.4.6,<0.5.0",
        "aioconsole>=0.6.1,<0.9.0",
        "accelerate>=0.26.0,<0.27.0"
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "llm-party-server=llm_party_chat.server:main",
            "llm-party-client=llm_party_chat.client:main",
            "llm-party-moderator=llm_party_chat.moderator:main",
        ],
    },
    author="Jim Beno",
    author_email="jim@jimbeno.net",
    description="Real-time chat between multiple LLMs with human moderator ability",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbeno/llm-party-chat",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)