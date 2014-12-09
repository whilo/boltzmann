(ns boltzmann.core
  (:require [boltzmann.protocols :refer :all]
            [boltzmann.formulas :as f]))


(def state-space f/state-space)

(defn energy
  "Energy of state x in Boltzmann Machine bm."
  [bm x]
  (f/energy (-weights bm) (-biases bm)))

(defn prob
  "Calculates the theoretical probability of the Boltzmann machine of state x.
  !!! o(exp), intractable for (smaller) layer sizes > 20 !!!"
  [bm x]
  (f/prob (-weights bm) (-biases bm) x))

(defn sample-gibbs [bm iterations & {:keys [start-state particles]
                                     :or {start-state (repeat (count (-biases bm)) 0)
                                          particles 1}}]
  (-sample-gibbs bm iterations start-state particles))

(defn train-cd [rbm batches & {:keys [epochs learning-rate k]
                               :or {epochs 1 learning-rate 0.1 k 1}}]
  (-train-cd rbm batches epochs learning-rate k))


(comment
  (require '[incanter.core :as i]
           '[incanter.datasets :as ds]
           '[incanter.charts :as c]
           '[clojure.core.matrix :as mat]
           '[boltzmann.theoretical :refer [create-theoretical-rbm]]
           '[boltzmann.jblas :refer [create-jblas-rbm]])

  (def test-rbm (create-theoretical-rbm [[1]] [2] [3]))
  (f/partition (-weights test-rbm) (-biases test-rbm))
  (reduce + (map #(boltz-prob test-rbm %) (state-space (count (-biases test-rbm)))))
  (sample-gibbs test-rbm 100)
  (mat/current-implementation)


  (time (let [weights [[0.5 0.1 0.3 -0.8]
                       [0.3 0.5 0.3 0.1]
                       [0.2 -0.8 -0.3 0.0]]
              v-bias [0.5 0.8 0.3 -0.7]
              h-bias [0.2 -0.3 0.0]
              rbm (create-theoretical-rbm weights v-bias h-bias)
              v-count (count v-bias)
              h-count (count h-bias)
              samples (mapv #(vec (take v-count %))
                            (sample-gibbs rbm 60000))
              histo (frequencies samples)

              all-states (state-space v-count)
                                        ;        probs (map #(-probability rbm %) all-states)
              probs (map #(f/prob weights v-bias %) all-states)

              batches (doall (map mat/matrix (partition 10 samples)))
              model (train-cd (create-theoretical-rbm v-count h-count) batches :epochs 30)
                                        ;             model (train-cd (create-jblas-rbm v-count h-count) batches :epochs 10)
              model-samples (mapv #(vec (take v-count %))
                                  (sample-gibbs model 60000))
              model-histo (frequencies model-samples)

              states (interleave all-states all-states all-states)
              probabilities (interleave probs
                                        (map #(/ % (count samples)) (map histo all-states))
                                        (map #(/ % (count model-samples)) (map model-histo all-states)))

              grouping (interleave (repeat "ideal") (repeat "sampled") (repeat "model"))]
          (i/view (c/bar-chart states probabilities
                               :group-by grouping
                               :legend true))))

  (def xor-states [[0 0 0] [0 1 1] [1 0 1] [1 1 1]])

  (mat/set-current-implementation :clatrix)

  (def xor-model
    (train-cd (create-jblas-rbm 3 2)
              (map (comp mat/matrix vector) xor-states)
              :epochs 5000))

  (let [v-count 3
        all-states (state-space v-count)
        model xor-model
        model-samples (mapv #(vec (take v-count %)) (sample-gibbs model 30000))
        model-histo (frequencies model-samples)
        probabilities (map #(/ % (count model-samples)) (map model-histo all-states))
        grouping (repeat "model")]
    (i/view (c/bar-chart all-states probabilities
                         :group-by grouping
                         :legend true)))

  (i/view (c/histogram (flatten (:restricted-weights xor-model))))
  (i/view (c/histogram (flatten (:v-biases xor-model))))
  (i/view (c/histogram (flatten (:h-biases xor-model))))

  (require '[boltzmann.jblas :refer [probs-hs-given-vs probs-vs-given-hs]])

  (->> xor-states
       (map (comp mat/matrix vector))
       (map #(->> %
                  (probs-hs-given-vs [(:restricted-weights xor-model)
                                      (:h-biases xor-model)])
                  (probs-vs-given-hs [(:restricted-weights xor-model)
                                      (:v-biases xor-model)])))
       (map (partial take 3)))
  )
