(ns boltzmann.protocols)

(defprotocol PBoltzmannMachine
  (-biases [this])
  (-weights [this]))

(defprotocol PRestrictedBoltzmannMachine
  (-v-biases [this])
  (-h-biases [this])
  (-restricted-weights [this]))

(defprotocol PContrastiveDivergence
  (-train-cd [this batches epochs learning-rate k prng back-ch])
  (-train-pcd [this states]))

(defprotocol PErrorEstimation
  (-cross-entropy [this activation-a activation-b])
  (-errors [this states labels]))

(defprotocol PSample
  (-sample-gibbs [this iterations start-state particles prng])
  (-sample-ast [this iterations start-state particles prng]))

(defprotocol PBackpropagation
  (-forward-prop [this state])
  (-train-backprop [this states]))
