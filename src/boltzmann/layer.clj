(ns boltzmann.layer
  "A layer datatype for one layer of hidden units and their weights to
  the visible ones."
  (:require [clojure.core.matrix :refer [matrix]]
            ;; TODO replace if possible
            [incanter.stats :refer [sample-normal]]))

(defrecord Layer [weights v-bias h-bias])

(defn create-layer [weights v-bias h-bias]
  (->Layer (matrix weights) (matrix v-bias) (matrix h-bias)))

(defn init-layer
  "Returns a layer consisting of weight matrix, visible and hidden bias.
  Weight matrix and hidden bias are initialized through a normal distribution
  around 0 with sd 0.01 to break symmetry."
  [v-count h-count]
  (create-layer (repeatedly h-count (fn [] (sample-normal v-count :mean 0 :sd 0.01)))
                [(vec (sample-normal v-count :mean 0 :sd 0.01))]
                [(vec (repeat h-count 0))]))
