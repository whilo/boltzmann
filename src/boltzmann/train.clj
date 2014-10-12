(ns boltzmann.train
  "Simple unoptimized implementations of contrastive divergence learning."
  (:require [boltzmann.layer :refer [create-layer init-layer]]
            [boltzmann.formulas :refer :all]
            [clojure.core.matrix :refer [exp log add sub mul matrix
                                         transpose columns get-row outer-product] :as mat]))

;; we use the fastest implementation of core.matrix for model sizes of 1000x1000
;; weight matrices, but persistent vector should always work for the simple training api
(mat/set-current-implementation :clatrix)

(defn probs-h-given-v [{:keys [weights v-bias h-bias]} v-state]
  (mapv (fn [i] (boltz-cond-prob weights h-bias v-state i))
        (range (count h-bias))))

(defn probs-v-given-h [{:keys [weights v-bias h-bias]} h-state]
  (mapv (fn [i] (boltz-cond-prob (transpose weights) v-bias h-state i))
        (range (count v-bias))))

(defn cd
  "Implements contrastive divergence with duration steps (CD-k),
  a duration (k) = 1 is often used and performs reasonably."
  [{:keys [weights v-bias h-bias] :as layer} v-data h-data duration]
  (let [v-count (count v-bias)
        h-count (count h-bias)
        init-chain [v-data h-data]]
    (reduce
     (fn [chain i]
       (let [v-recon (probs-v-given-h layer (sample-binary (get chain (dec (count chain)))))]
         (-> chain
             (conj v-recon)
             (conj (probs-h-given-v layer (sample-binary v-recon))))))
     init-chain
     (range duration))))

(defn calc-up
  "Calculate the total updates on the model (usually calculated through a cd chain,
  approximating <v-data,h-data> - <v-model,h-model>. "
  [v-model h-model v-data h-data]
  (create-layer (sub (outer-product h-data v-data)
                     (outer-product h-model v-model))
                (sub v-data v-model)
                (sub h-data h-model)))

(defn train-cd [layer data rate]
  (reduce (fn [{:keys [weights v-bias h-bias] :as layer} v-probs]
            (let [v-data (sample-binary v-probs) ;
                  h-probs (probs-h-given-v layer v-data)
                  h-data (sample-binary h-probs)
                  chain (cd layer v-data h-data 1)
                  up (calc-up (get chain (- (count chain) 2))
                               (get chain (dec (count chain)))
                               v-probs h-probs)]
              (merge-with #(add %1 (mul %2 rate))
                          layer
                          up)))
          layer
          data))


(comment
  (require '[incanter.core :as i]
           '[clojure.math.combinatorics :refer [cartesian-product]]
           '[incanter.datasets :as ds]
           '[incanter.charts :as c])

  ;; do a simple (non-batch) test with a small state space, so we can compare to the ideal distribution:
  (let [weights [[0.5 0.1 0.3 -0.8]
                 [0.3 0.5 0.3 0.1]]
        v-bias [0.5 0.8 0.3 -0.7]
        h-bias [0.2 -0.3]
        v-count (count v-bias)
        h-count (count h-bias)
        samples (mapv #(vec (take (count v-bias) %))
                      (gibbs-sampler (full-matrix weights)
                                     (vec (concat v-bias h-bias))
                                     10000))
        histo (reduce #(update-in %1 [%2] (fnil inc 0)) {} samples)

        eta 0.001
        all-states (mapv vec (apply cartesian-product (repeat v-count [0 1])))
        probs (map #(boltz-prob weights v-bias all-states %)
                   all-states)

        model (reduce (fn [model step]
                        (train-cd model samples (/ eta step)))
                      (init-layer v-count h-count)
                      (range 1 20))
        {w* :weights vb* :v-bias hb* :h-bias} model
        model-samples (mapv #(vec (take v-count %))
                            (gibbs-sampler (full-matrix w*)
                                           (vec (concat vb* hb*))
                                           10000))
        model-histo (reduce #(update-in %1 [%2] (fnil inc 0)) {} model-samples)

        states (interleave all-states all-states all-states)
        probabilities (interleave probs
                                  (map #(/ % (count samples)) (map histo all-states))
                                  (map #(/ % (count model-samples)) (map model-histo all-states)))
        grouping (interleave (repeat "ideal") (repeat "sampled") (repeat "model"))]
    (i/view (c/bar-chart states probabilities
                         :group-by grouping
                         :legend true)))






  ;; from hy-lang (python) only for reference
  (let [[weights [[0.0 0.0 0.5 0.3]
                  [0.0 0.0 0.1 0.5]
                  [0.5 0.1 0.0 0.0]
                  [0.3 0.5 0.0 0.0]]]
        [biases [0.5 0.8 0.2 -0.3]]
        [test (gibbs-sampler boltz-cond-prob weights biases (create-chain [0 0 0 0]) 1)]
        [data (list (map (fn [e] (array (list (take 2 e)))) test))]
        [avg-data (/ (sum data)
                     (len data))]
        [w1 [[0.5 0.1]
             [0.3 0.5]]]
        [b1 [0.5 0.8]]
        [b2 [0.2 -0.3]]
        [v-data (rand-nth data)]
        [M (array (repeat-star (list (map (fn [i] (/ (.uniform random) 100))
                                          (range 2)))
                               2))]
        [v-bias (array (list (map (fn [p_i]
                                    (log (/ p_i
                                            (- 1 p_i))))
                                  avg-data)))]
        [h-bias (zeros 2)
                                        ;      (array (list (map (fn [i] (/ (.uniform random) 10))
                                        ;                               (range 2))))
         ]
        [eta 0.01]]
    (reduce (fn [model step]
              (print (+ "step: " (str step)))
              (let [[[weights v-bias h-bias] model]
                    [up (cd weights v-bias h-bias (rand-nth data) 1)]]
                (print "up")
                (print up)
                (print "after up")
                (print (+ model (* eta (array up))))
                (+ model (* eta (array up)))))
            (range 10)
            [M v-bias h-bias]))


  ;; bar-chart api samples

  (i/view (c/bar-chart (mapcat (fn [x] [x x]) [[0 0] [0 1] [1 0] [1 1]])
                       '(0.1 0.3 0.4 0.2 0.4 0.2 0.1 0.3)
                       :group-by (interleave (repeat "s") (repeat "i"))))

  (i/view (c/bar-chart ["a" "a" "b" "b" "c" "c" ] [10 20 30 10 40 20]
                       :legend true
                       :group-by ["I" "II" "I" "II" "I" "II"]))




  (i/$rollup :count :foo :bar (i/to-dataset [{:foo 1 :bar 2 :baz 3}
                                             {:foo 4 :bar 5 :baz 6}]))



  (def data (ds/get-dataset :airline-passengers))
  (i/view (c/bar-chart :year :passengers :group-by :month :legend true :data data)))
