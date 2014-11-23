(ns boltzmann.sample)

(defn sample-binary
  "Samples a vector of binary values from a vector of probabilities."
  [probabilities]
  (mapv (fn [p] (if (< (rand) p) 1 0)) probabilities))
