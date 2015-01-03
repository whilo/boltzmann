(ns boltzmann.sample)

(defn sample-binary
  "Samples a vector of binary values from a vector of probabilities."
  [prng probabilities]
  (mapv (fn [p] (if (< (prng) p) 1 0)) probabilities))
