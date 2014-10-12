(ns boltzmann.batch-train
  "Optimized routines on mini-batches of training samples. The code is inlined with JBlas in critical parts and type-hinted."
  (:require [boltzmann.layer :refer [create-layer init-layer]]
            [boltzmann.batch-formulas :refer :all]
            [clatrix.core :refer [->Matrix]]
            [clojure.core.matrix :refer [add sub mul matrix transpose columns get-row] :as mat])
  (:import [org.jblas MatrixFunctions DoubleMatrix]
           [clatrix.core Matrix]))


(defn sample-binary-matrix [m]
  (let [[x y] (mat/shape m)]
    (->Matrix (.ge (.-me m) (DoubleMatrix/rand x y))
              {})))


(defn probs-hs-given-vs [{:keys [^Matrix weights h-bias]} v-states]
  (boltz-cond-prob-batch weights h-bias v-states))


(defn probs-vs-given-hs [{:keys [^Matrix weights v-bias]} h-states]
  (boltz-cond-prob-batch (->Matrix (.transpose ^DoubleMatrix (.-me weights))
                                   {})
                         v-bias h-states))


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
                           sample-binary-matrix
                           (probs-vs-given-hs layer))]
         (-> chain
             (conj v-recons)
             (conj (->> v-recons
                        sample-binary-matrix
                        (probs-hs-given-vs layer) )))))
     init-chain
     (range duration))))


;; TODO dedup, atm. outer-product needs DoubleMatrix types for performance
(defn calc-batch-up
  "Calculate the total updates on the model (usually calculated through a cd chain,
  approximating <v-data,h-data> - <v-model,h-model>. "
  [v-model h-model v-data h-data]
  ;; jvisualvm: current 9/10 overhead in cd learning mapv
  (create-layer (sub (outer-product h-data v-data)
                     (outer-product h-model v-model))
                (sub v-data v-model)
                (sub h-data h-model)))


(defn train-cd-batch [layer batches rate]
  (reduce (fn [layer v-probs-batch]
            (let [v-data-batch (sample-binary-matrix v-probs-batch)
                  h-probs-batch (probs-hs-given-vs layer v-data-batch)
                  chain (cd-batch layer v-probs-batch h-probs-batch 1)
                  up (calc-batch-up (apply add (get chain (- (count chain) 2)))
                                    (apply add (get chain (dec (count chain))))
                                    (apply add v-probs-batch)
                                    (apply add h-probs-batch))]
              (merge-with #(add %1 (mul %2 (/ rate (count v-probs-batch))))
                          layer
                          up)))
          layer
          batches))


(comment
  ;; require some more stuff for live coding, should not be in the library
  (require '[boltzmann.mnist :as mnist]
           '[criterium.core :refer [bench]])

  (mat/set-current-implementation :persistent-vector)
  (mat/current-implementation)

  (future (def images (mnist/read-images "resources/train-images-idx3-ubyte"))
          (def batches (doall (map matrix (partition 10 images)))))

  (bench (train-cd-batch (init-layer 784 1000) (take 10 batches) 0.1) :verbose)


  ;; investigate theano error fn

  (def labels (mnist/read-labels "resources/train-labels-idx1-ubyte"))


  (def trained
    (let [il (init-layer 784 1000)]
      #_(->> (first batches)
             (probs-hs-given-vs il)
             (map sample-binary)
             (probs-vs-given-hs il)
             time)
      (time (train-cd-batch il (take 100 batches) 0.1))))

  (->> images
       (take 100)
       (map #(->> % (probs-h-given-v trained) (probs-v-given-h trained)))
       (map #(partition 28 %))
       (mnist/tile 10)
       mnist/render-grayscale-float-matrix
       mnist/view)

  (require '[boltzmann.formulas :refer [full-matrix boltz-prob gibbs-sampler]]
           '[clojure.math.combinatorics :refer [cartesian-product]]
           '[incanter.core :as i]
           '[incanter.charts :as c])
  ;; test batch fns

  (let [weights [[0.5 0.1 0.3 -0.8]
                 [0.3 0.5 0.3 0.1]]
        v-bias [0.5 0.8 0.3 -0.7]
        h-bias [0.2 -0.3]
        v-count (count v-bias)
        h-count (* 5 (count h-bias))
        samples (mapv #(vec (take (count v-bias) %))
                      (gibbs-sampler (full-matrix weights)
                                     (vec (concat v-bias h-bias))
                                     20000))
        histo (reduce #(update-in %1 [%2] (fnil inc 0)) {} samples)

        eta 0.001
        all-states (mapv vec (apply cartesian-product (repeat v-count [0 1])))
        probs (map #(boltz-prob weights v-bias all-states %)
                   all-states)

        model (reduce (fn [model step]
                        (train-cd-batch model
                                        (map matrix (partition 10 samples))
                                        (/ eta step)))
                      (init-layer v-count h-count)
                      (range 1 10))
        {w* :weights vb* :v-bias hb* :h-bias} model
        model-samples (mapv #(vec (take v-count %))
                            (gibbs-sampler (full-matrix w*)
                                           (vec (concat vb* hb*))
                                           20000))
        model-histo (reduce #(update-in %1 [%2] (fnil inc 0)) {} model-samples)

        states (interleave all-states all-states all-states)
        probabilities (interleave probs
                                  (map #(/ % (count samples)) (map histo all-states))
                                  (map #(/ % (count model-samples)) (map model-histo all-states)))
        grouping (interleave (repeat "ideal") (repeat "sampled") (repeat "model"))]
    (i/view (c/bar-chart states probabilities
                         :group-by grouping
                         :legend true))))
