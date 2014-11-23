(ns boltzmann.matrix
  (:require [clojure.core.matrix :refer [transpose zero-matrix join-along]]))

(defn full-matrix
  "Expands a matrix from a restricted Boltzmann machine, which only
  covers visible and hidden units, into a symmetric matrix with zero
  blocks between units of each layer. This is used to use the normal
  gibbs-sampler to sample from the full Boltzmann machine."
  [visi-hidden-matrix]
  (let [v-count (count (first visi-hidden-matrix))
        h-count (count visi-hidden-matrix)]
    (join-along 0 ;; append to bottom
                (join-along 1 ;; append to right
                            (zero-matrix v-count v-count)
                            (transpose visi-hidden-matrix))
                (join-along 1
                            visi-hidden-matrix
                            (zero-matrix h-count h-count)))))
