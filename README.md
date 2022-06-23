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
    <a href="https://github.com/tmrob2/motap-hdd">View Demo</a>
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

This project is an API implemented in Rust v1.5 targetting very large state spaces, possible millions 
of states, which do not fit into memory. Agents consist of constructing an Markov decision process (MDP) 
environment, and tasks consist of deterministic finite automata (DFA) which correspond to co-safe linear 
temporal logic (LTL). Examples and details on how to install the project dependencies are expained below. 
The project is suffixed with hdd because it stores partitions of a problem state-space on the hard-disk.
<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Rust v1.61](https://www.rust-lang.org/)
* [SuiteSparse - CXSparse v5.8](https://github.com/DrTimothyAldenDavis/SuiteSparse)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The MOTAP-hdd solver uses a sparse linear algebra library written in C called CXSparse. A small FFI C 
wrapper is pre-built for this library in src/c_bindings/

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/tmrob2/motap-hdd.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

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

Distributed under the MIT License. See `LICENSE.txt` for more information.

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
