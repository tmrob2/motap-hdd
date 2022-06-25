<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->
[![Issues][issues-shield]][issues-url]
[![Apache License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/tmrob2/motap-hdd">
    <!--<img src="images/logo.png" alt="Logo" width="80" height="80">-->
  </a>

<h3 align="center">Multi-objective Task Allocation and Planning (MOTAP)</h3>

  <p align="center">
    Considers the simultaneous task allocation planning of multiagent systems. 
    <br />
    <a href="https://github.com/tmrob2/motap-hdd"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/tmrob2/rusty-robots">View Demo</a>
    ·
    <a href="https://github.com/tmrob2/motap-hdd/issues">Report Bug</a>
    ·
    <a href="https://github.com/tmrob2/motap-hdd/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

This project is a <span style="text-decoration: underline">prototype</span> API implemented in Rust v1.5 targetting very large state spaces, possible millions 
of states, which do not fit into memory.
Agents consist of constructing an Markov decision process (MDP) environment, and tasks consist of deterministic finite automata (DFA) which correspond to co-safe linear 
temporal logic (LTL). Examples and details on how to install the project dependencies are expained below. 
The project is suffixed with hdd because it stores partitions of a problem state-space on the hard-disk.
This project is a part of work into efficient improvements in centralised task allocation in 
multiagent systems.

Developed for Linux.
<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Rust v1.61](https://www.rust-lang.org/)
* [SuiteSparse - CXSparse v5.11](https://github.com/DrTimothyAldenDavis/SuiteSparse/CXSparse)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The MOTAP-hdd solver uses a sparse linear algebra library written in C called CXSparse. A small FFI C 
wrapper is pre-built for this library in src/c_binding/suite_sparse.rs. Only the library dependencies
will need to be configured in build.rs. Installing a Sparse linear algebra library can be a bit tricky
but it by far produces the best results for model solving speed.

### Prerequisites

#### SparseSuite CXSparse v5.11 - Installation

First clone the SuiteSparse project from https://github.com/DrTimothyAldenDavis/SuiteSparse.
To install CXSparse, first edit 
```
SuiteSparse/SuiteSparse_config/SuiteSparse_config.mk
```
with your required configuration settings. This will include a library link to your favourite BLAS instance
but Intel MKL is recommended for SuiteSparse.

There are a lot of different Sparse computation methods in this project, but the only one we require is CXSparse.
Check your current configuration using the following in the root directory of SuiteSparse. The main thing required
is a dependency to a BLAS library. 
```sh
make config
```
If MKLROOT is detected this will be the BLAS implementation, otherwise specify the shared library for BLAS with -lblas. 
Goto the suitsparse directory and install with 
```shell
make 
make install # constructs shared libraries which our build.rs script will connect to. 
```
Testing if CSSparse was installed correctly can be done after the project has been built.

#### Rust v1.61 - Installation
To install Rust follow the instructions and pre-requisites at goto https://www.rust-lang.org/learn/get-started. Usually 
this is as simple as running the listed command in your terminal. 

Also required is Cargo the Rust package manager. There are other ways of compiling Rust projects but
Cargo is by far the easiest.

#### Gurobi
Certain calculations for computing vectors to approximate a Pareto curve rely on a good linear programming 
software. Gurobi is used as the LP solver in this package. Gurobi is a subscription based software but
can also be used under an academic license. 

Download an install Gurobi Optimizer at https://www.gurobi.com/downloads/. Once installed
the Rust Gurobi wrapper crate of this project will automatically bind to this.

### Installation

Installation of the MOTAP tool itself is simple. Once this repository is cloned you can run 
```sh
cargo build
``` 
This may take a few minutes to compile the first time as a CBLAS instance will need to be installed. This is 
automatically configured for your system.  

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

An example implementation on how to use this crate can be found at https://github.com/tmrob2/rusty-robots.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/tmrob2/motap-hdd/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/tmrob2/motap-hdd](https://github.com/tmrob2/motap-hdd)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/tmrob2/motap-hdd.svg?style=for-the-badge
[contributors-url]: https://github.com/tmrob2/motap-hdd/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/tmrob2/motap-hdd.svg?style=for-the-badge
[forks-url]: https://github.com/tmrob2/motap-hdd/network/members
[stars-shield]: https://img.shields.io/github/stars/tmrob2/motap-hdd.svg?style=for-the-badge
[stars-url]: https://github.com/tmrob2/motap-hdd/stargazers
[issues-shield]: https://img.shields.io/github/issues/tmrob2/motap-hdd.svg?style=for-the-badge
[issues-url]: https://github.com/tmrob2/motap-hdd/issues
[license-shield]: https://img.shields.io/github/license/tmrob2/motap-hdd.svg?style=for-the-badge
[license-url]: https://github.com/tmrob2/motap-hdd/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
