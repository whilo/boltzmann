(ns boltzmann.restricted
  (:require [boltzmann.formulas :refer :all]
            [clojure.core.matrix :refer [exp log add sub mul matrix transpose columns] :as mat]
            [clojure.math.combinatorics :refer [cartesian-product]]
            ;; TODO replace if possible
            [incanter.stats :refer [sample-normal]]))

;; avoid warnings for now in 1.7, but doesn't work
;(set! clojure.core/*unchecked-math* true)

(defrecord Layer [weights v-bias h-bias])

(defn sample-binary [probabilities]
  (mapv (fn [p] (if (< (rand) p) 1 0)) probabilities))

(defn boltz-gibbs-sampler [w bs iterations]
  (reduce (fn [chain step]
            (let [last (get chain (dec (count chain)))]
              (conj chain (->> (range (count bs))
                               (map (partial boltz-cond-prob w bs last))
                               sample-binary))))
          [(repeat (count bs) 0)] ;; init
          (range iterations)))

(defn probs-h-given-v [{:keys [weights v-bias h-bias]} v-state]
  (mapv (fn [i] (boltz-cond-prob weights h-bias v-state i))
        (range (count h-bias))))

(defn prepare-batch [states]
  (mat/join-along 1 (repeat [1]) states))

(defn probs-hs-given-vs [{:keys [weights h-bias]} v-states]
  (boltz-cond-prob-batch (mat/join-along 1
                                         (map vector h-bias)
                                         weights)
                         v-states))

(defn probs-v-given-h [{:keys [weights v-bias h-bias]} h-state]
  (mapv (fn [i] (boltz-cond-prob (transpose weights) v-bias h-state i))
        (range (count v-bias))))

(defn probs-vs-given-hs [{:keys [weights v-bias]} h-states]
  (boltz-cond-prob-batch (mat/join-along 1
                                         (map vector v-bias)
                                         (transpose weights))
                         h-states))

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

(defn cd-batch
  "Implements contrastive divergence with duration steps (CD-k),
  a duration (k) = 1 is often used and performs reasonably."
  [layer v-data h-data duration]
  (let [v-count (count v-data)
        h-count (count h-data)
        init-chain [v-data h-data]]
    (reduce
     (fn [chain i]
       (let [last (get chain (dec (count chain)))
             v-recons (->> last
                           (map sample-binary)
                           prepare-batch
                           (probs-vs-given-hs layer))]
         (-> chain
             (conj v-recons)
             (conj (->> v-recons
                        (map sample-binary)
                        prepare-batch
                        (probs-hs-given-vs layer) )))))
     init-chain
     (range duration))))

(defn calc-up
  "Calculate the total updates on the model (usually calculated through a cd chain,
  approximating <v-data,h-data> - <v-model,h-model>. "
  [v-model h-model v-data h-data]
  (->Layer (matrix (mapv #(sub (mul v-data %1)
                               (mul v-model %2)) h-data h-model))
           (sub v-data v-model)
           (sub h-data h-model)))

(defn init-layer
  "Returns a layer consisting of weight matrix, visible and hidden bias.
  Weight matrix and hidden bias are initialized through a normal distribution
  around 0 with sd 0.01 to break symmetry."
  [v-count h-count]
  (->Layer (matrix (repeatedly h-count (fn [] (sample-normal v-count :mean 0 :sd 0.01))))
           (vec (sample-normal v-count :mean 0 :sd 0.01))
           (vec (repeat h-count 0))))


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

(defn train-cd-batch [layer batches rate]
  (reduce (fn [layer v-probs-batch]
            (let [v-data-batch (map sample-binary v-probs-batch)
                  h-probs-batch (->> v-data-batch
                                     prepare-batch
                                     (probs-hs-given-vs layer))
                  h-data-batch (map sample-binary h-probs-batch)
                  chain (cd-batch layer v-data-batch h-data-batch 1)
                  up (calc-up (apply mat/add (get chain (- (count chain) 2)))
                               (apply mat/add (get chain (dec (count chain))))
                               (apply mat/add v-probs-batch)
                               (apply mat/add h-probs-batch))]
              (merge-with #(add %1 (mul %2 (/ rate (count v-probs-batch))))
                          layer
                          up)))
          layer
          batches))


(comment
  ;; require some more stuff for live coding, should not be in the library
  (require '[boltzmann.mnist :as mnist]
           '[criterium.core :refer [bench]]
           '[incanter.core :as i]
           '[incanter.datasets :as ds]
           '[incanter.charts :as c])

  (def images (mnist/read-images "resources/train-images-idx3-ubyte"))

  (def batches (doall (partition 10 images)))

  (bench (train-cd-batch (init-layer 784 1000) (take 10 batches) 0.1))

  (def labels (mnist/read-labels "resources/train-labels-idx1-ubyte"))


  ;; test batch fns
  (let [weights [[0.5 0.1 0.3 -0.8]
                 [0.3 0.5 0.3 0.1]]
        v-bias [0.5 0.8 0.3 -0.7]
        h-bias [0.2 -0.3]
        layer (Layer. weights v-bias h-bias)
        v-count (count v-bias)
        h-count (count h-bias)
        samples (mapv #(vec (take (count v-bias) %))
                      (boltz-gibbs-sampler (full-matrix weights)
                                           (vec (concat v-bias h-bias))
                                           10000))
        histo (reduce #(update-in %1 [%2] (fnil inc 0)) {} samples)

        eta 0.001
        all-states (mapv vec (apply cartesian-product (repeat v-count [0 1])))
        probs (map #(boltz-prob weights v-bias all-states %)
                   all-states)

        model (reduce (fn [model step]
                        (train-cd-batch model
                                        (partition 10 samples)
                                        (/ eta step)))
                      (init-layer v-count h-count)
                      (range 1 10))
        {w* :weights vb* :v-bias hb* :h-bias} model
        model-samples (mapv #(vec (take v-count %))
                            (boltz-gibbs-sampler (full-matrix w*)
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


  ;; do a simple (non-batch) test with a small state space, so we can compare to the ideal distribution:
  (let [weights [[0.5 0.1 0.3 -0.8]
                 [0.3 0.5 0.3 0.1]]
        v-bias [0.5 0.8 0.3 -0.7]
        h-bias [0.2 -0.3]
        layer (Layer. weights v-bias h-bias)
        v-count (count v-bias)
        h-count (count h-bias)
        samples (mapv #(vec (take (count v-bias) %))
                      (boltz-gibbs-sampler (full-matrix weights)
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
                      (range 1 10))
        {w* :weights vb* :v-bias hb* :h-bias} model
        model-samples (mapv #(vec (take v-count %))
                            (boltz-gibbs-sampler (full-matrix w*)
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
  (i/view (c/bar-chart :year :passengers :group-by :month :legend true :data data))

  (mat/set-current-implementation :clatrix)
  (mat/set-current-implementation :persistent-vector)
  (mat/current-implementation))
