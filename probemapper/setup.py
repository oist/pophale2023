from setuptools import setup, find_packages

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
	name="probemapper",
	version="0.0.1",
	author="Tomoyuki Mano",
	author_email="tomoyuki.mano1@oist.jp",
	description="",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="",
	packages=find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	]
)
