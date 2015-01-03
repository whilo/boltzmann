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

(defn sample-gibbs [bm iterations & {:keys [start-state particles seed]
                                     :or {start-state (repeat (count (-biases bm)) 0)
                                          particles 1
                                          seed 42}}]
  (-sample-gibbs bm iterations start-state particles seed))

(defn train-cd [rbm batches & {:keys [epochs learning-rate k seed]
                               :or {epochs 1 learning-rate 0.01 k 1 seed 42}}]
  (-train-cd rbm batches epochs learning-rate k seed))


(comment
  (require '[incanter.core :as i]
           '[incanter.datasets :as ds]
           '[incanter.charts :as c]
           '[clojure.core.matrix :as mat]
           '[boltzmann.theoretical :refer [create-theoretical-rbm]]
           '[boltzmann.jblas :refer [create-jblas-rbm]])

  (f/partition (-weights test-rbm) (-biases test-rbm))
  (reduce + (map #(boltz-prob test-rbm %) (state-space (count (-biases test-rbm)))))
  (sample-gibbs test-rbm 100)
  (mat/current-implementation)

  (require '[clojure.data.json :as json])


  (mat/set-current-implementation :clatrix)
  (time (let [weights [[0.5 -0.2 0.3]
                       [-0.2 -0.4 0.2]]
              #_[[0.05 0.01 0.0]
                 [0.03 -0.05 -0.01]
                 [0.02 -0.08 0.03]]
              v-bias [0.0 0.0 0.0]
              h-bias [0.0 0.0]
              rbm (create-theoretical-rbm weights v-bias h-bias)
              v-count (count v-bias)
              h-count (count h-bias)
              samples (mapv #(vec (take v-count %))
                            (sample-gibbs rbm 10000))
              histo (frequencies samples)

              all-states (state-space v-count)
                                        ;        probs (map #(-probability rbm %) all-states)
              probs (map #(f/prob weights v-bias %) all-states)

                                        ;              model (train-cd (create-theoretical-rbm v-count (* 3 h-count)) samples :epochs 30)
              batches (doall (map mat/matrix (partition 1 samples)))
              model (train-cd (create-jblas-rbm v-count (* 2 h-count)) batches :epochs 1)
              model-samples (mapv #(vec (take v-count %))
                                  (sample-gibbs model 80000))
              model-histo (frequencies model-samples)

              states (interleave all-states all-states all-states)
              probabilities (interleave probs
                                        (map #(/ % (count samples)) (map histo all-states))
                                        (map #(/ % (count model-samples)) (map model-histo all-states)))

              grouping (interleave (repeat "ideal") (repeat "sampled") (repeat "model"))]
          (i/view (c/bar-chart states probabilities
                               :group-by grouping
                               :legend true))
          (i/save (c/bar-chart states probabilities
                               :group-by grouping
                               :legend true)
                  "/tmp/small-distribution.png")
          (i/save (c/histogram (flatten (:restricted-weights model))) "/tmp/small-weights.png" :width 1000)
          (i/save (c/histogram (flatten (:v-biases model))) "/tmp/small-v-biases.png" :width 1000)
          (i/save (c/histogram (flatten (:h-biases model))) "/tmp/small-h-biases.png" :width 1000)

          (spit "/tmp/samples.json" (json/write-str samples))
          (def test-rbm model)))

  (time (let [weights [[0.5 -0.2 0.3]
                       [-0.2 -0.4 0.2]]
              v-bias [0.5 0.8 0.3]
              h-bias [0.2 -0.3 0.0]
              rbm (create-theoretical-rbm weights v-bias h-bias)
              v-count (count v-bias)
              h-count (count h-bias)
              samples (json/read-str (slurp "/tmp/samples.json"))
              histo (frequencies samples)

              all-states (state-space v-count)
                                        ;        probs (map #(-probability rbm %) all-states)
              probs (map #(f/prob weights v-bias %) all-states)

                                        ;              model (train-cd (create-theoretical-rbm v-count (* 3 h-count)) samples :epochs 30)
              batches (doall (map mat/matrix (partition 1 samples)))
              model (train-cd (create-theoretical-rbm v-count (* 2 h-count)) batches :epochs 1)
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
                               :legend true))
          (i/save (c/bar-chart states probabilities
                               :group-by grouping
                               :legend true)
                  "/tmp/small-distribution.png" :width 1000)
          (i/save (c/histogram (flatten (:restricted-weights model))) "/tmp/small-weights.png" :width 1000)
          (i/save (c/histogram (flatten (:v-biases model))) "/tmp/small-v-biases.png" :width 1000)
          (i/save (c/histogram (flatten (:h-biases model))) "/tmp/small-h-biases.png" :width 1000)


          (def test-rbm model)))

  (def xor-states (take 20000 (cycle [[0 0 0] [0 1 1] [1 0 1] [1 1 0]])))

  (mat/set-current-implementation :clatrix)

  (def xor-model
    (train-cd (create-jblas-rbm 3 2)
              (map (comp mat/matrix vector) xor-states)
              :epochs 20))

  ;; TODO batch size 3?
  (let [v-count 3
        all-states (state-space v-count)
        model xor-model
        model-samples (mapv #(vec (take v-count %)) (sample-gibbs model 100000))
        model-histo (frequencies model-samples)
        probabilities (map #(/ (or % 0) (count model-samples)) (map model-histo all-states))
        grouping (repeat "model")]
    #_(i/view (c/bar-chart all-states probabilities
                           :group-by grouping
                           :legend true))
    (i/save (c/bar-chart all-states probabilities
                         :group-by grouping
                         :legend true)
            "/tmp/state-distribution.png"
            :width 1000))

  (i/view (c/histogram (flatten (:restricted-weights xor-model))))
  (i/view (c/histogram (flatten (:v-biases xor-model))))
  (i/view (c/histogram (flatten (:h-biases xor-model))))

  (i/save (c/histogram (flatten (:restricted-weights xor-model))) "/tmp/weights.png" :width 1000)
  (i/save (c/histogram (flatten (:v-biases xor-model))) "/tmp/v-biases.png" :width 1000)
  (i/save (c/histogram (flatten (:h-biases xor-model))) "/tmp/h-biases.png" :width 1000)

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
