# ECE 461/561 - Mini Projects

## Mini Project \#1: Two-Transmission-Link Queueing System Simulator and Output

This is a project to simulate a two link M/M/1/k queueing system in various
configuration formats. It ouputs various metrics and graphs to help visualize
differences in configurations. **Reports for this project in both PDF and HTML
format can be found in the `./docs` folder.**

***

## Mini Project \#2: Simulation and Analysis of a Circuit-Switched Optical Network with Wavelength-Division Multiplexing

This is a project to simulate a circuit-switched optical network with wavelength-division multiplexing. It outputs dropping probabilities for various network configurations for comparison. **Reports for this project in both PDF and HTML format can be found in the `./docs` folder.**
=======
This is a project to simulate a two link M/M/1/k queueing system in various configuration formats. It outputs various metrics and graphs to help visualize differences in configurations. **Reports for the project in both PDF and HTML types can be found in the `./docs` folder.**
>>>>>>> 65eb9e6a1469aa81926ec29ff8f2aa21e1929ff5

***

## Installation

Must have Python3 installed along with the SciPy stack found [here](https://www.scipy.org/install.html#installing-via-pip). If those are installed clone the repository: `git clone https://github.com/dlubawy/Network-Simulations.git <install directory>`. Replace `<install directory>` with where you want the project.

***

## Usage

Change directory to where you downloaded the files. Start a Jupyter notebook with the command `jupyter notebook`. This will start a web-server at `localhost:8888`, connect and login following the instructions given. The files should show up in your notebook, if not you can just upload them using the interface. To run the simulation just go to the `*.ipynb` file and run the kernel. Optionally, you may run just the simulation using `jupyter run docs/*.py`. This will run just the simulation code.

***

## License

Network Simulations

Copyright 2017 Andrew Lubawy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
