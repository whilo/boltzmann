# boltzmann

This library is supposed to implement [Boltzmann Machine](https://en.wikipedia.org/wiki/Boltzmann_machine) related /deep learning technologies/. All implementations should both have a clean high-level mathematical implementation of their algorithms and if possible, an optimized and benchmarked version of the core routines for production use. This is to facilitate learning for new users or potential contributors, to be able to implement algorithms from papers/other languages and then tune them for performance if needed.

This repository is supposed to cover techniques building on [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine), like [Deep Belief Networks](https://en.wikipedia.org/wiki/Deep_belief_network), [Deep Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf) or temporal extensions thereof. Classical back-propagation is also used to fine-tune deep models supervised, so it might be covered as well, but probably factored to a separate library then.

For other deep learning related techniques like autoencoders, a seperate library was reasonable, although some general routines might be shared through a library then. The idea is that you can include specific machine learning algorithms without pulling a whole machine learning framework into your project. This is a state-of-the-art [reference for deep learning in general in Python](http://deeplearning.net/tutorial/). Feel free to port examples to this library as an exercise and open pull-requests :).

## Usage

TODO put on clojars and demonstrate some learning.

## TODO

- track reconstruction error
- train some mnist and add model + performance table
- implement PCD
- implement more plots & visualize training of mnist
- do benchmarking (esp. of matrix ops) and algorithmic optimizations
- implement/port AIS, CAST
- make incanter dev dependency
- add python version for comparison?

## License

Copyright © 2014 Christian Weilbach

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
