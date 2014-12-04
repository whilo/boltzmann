# boltzmann

This library is supposed to implement [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine), Autoencoders and related deep learning technologies. All implementations should both have a clean high-level mathematical implementation of their algorithms (with `core.matrix`) and if possible, an optimized and benchmarked version of the core routines for production use. This is to facilitate learning for new users or potential contributors, to be able to implement algorithms from papers/other languages and then tune them for performance if needed.

This repository is supposed to cover techniques building on [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine), like [Deep Belief Networks](https://en.wikipedia.org/wiki/Deep_belief_network), [Deep Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf) or temporal extensions thereof as well as Autoencoders (which I am not familiar enough with yet). Classical back-propagation is also often used to fine-tune deep models supervisedly, so networks should support it as well.

This is a state-of-the-art [reference for deep learning in general in Python](http://deeplearning.net/tutorial/). Feel free to port examples to this library as an exercise and open pull-requests :).

## Usage

Demonstrate mnist learning.

## TODO

- put on clojars and demonstrate some learning.
- introduce fixed seeds
- track reconstruction error asynchronously and allow it to control/finish training
- factor out mnist (norb etc.) data set learning for others
- implement PCD
- implement more plots & visualize training of mnist
- do benchmarking (esp. of matrix ops) and algorithmic optimizations:
  investigate https://github.com/fommil/netlib-java/ with mtj, for 1000x1000 matrices 10x speedup on cpu vs. jBLAS on google mmul benchmark; investigate jcublas for gpu
- implement/port AIS, CAST, AST sampling
- make incanter dev dependency
- try to hide from skynet ;-)

## License

Copyright © 2014 Christian Weilbach

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
