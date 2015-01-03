(ns boltzmann.jblas
  "Optimized code that is inlined with JBlas in critical parts and type-hinted."
  (:require [boltzmann.protocols :refer :all]
            [boltzmann.matrix :refer [full-matrix]]
            [boltzmann.formulas :refer [cond-prob]] ;; TODO remove with batch sampling
            [boltzmann.sample :refer [sample-binary]]
            [clatrix.core :refer [->Matrix]]
            [clojure.core.matrix :refer [add sub mul matrix transpose columns get-row zero-matrix rows] :as mat]
            [incanter.stats :refer [sample-normal]])
  (:import [org.jblas MatrixFunctions DoubleMatrix]
           [clatrix.core Matrix]))

;; we use the fastest implementation of core.matrix for model sizes of 1000x1000
;; weight matrices, but persistent vector should always work for the simple training api
;; Dropout!, Rectified Linear Units https://www.youtube.com/watch?v=vShMxxqtDDs, Convolution when spatial input data
;; Alex Graves, online hand-writing with recurrent deep neural net
;; Ilya Sutskever model to learn language/english wikipedia
;; Layering of recurrent model
(mat/set-current-implementation :clatrix)


(defn outer-product [^Matrix a ^Matrix b]
  "Outer product (between two vectors given as 1-dimensional matrices)."
  (let [ac (count a)
        bc (count b)]
    (->Matrix (.rankOneUpdate ^DoubleMatrix (.-me (zero-matrix ac bc))
                              ^DoubleMatrix (.-me a)
                              ^DoubleMatrix (.-me b))
              {})))

(defn cond-prob-batch                   ;
  "Calculate the activations of a whole batch of training samples for
  an averaged update in CD. This is in general done because matrix
  multiplications are highly optimized on standard cpus and gpus.

  This routine has currently clatrix/JBlas inlined."
  [^Matrix w ^Matrix bias ^Matrix batch]
  (-> (.mmul ^DoubleMatrix (.dup (.-me w)) ^DoubleMatrix (.transpose (.-me batch)))
      .transpose
      ;; implement sigmoid as matrix op through exp
      (.addRowVector (.getRow (.-me bias) 0))
      (.mul -1.0)
      MatrixFunctions/exp
      (.add +1.0)
      (.rdiv 1.0)
      (->Matrix {})))


(defn sample-binary-matrix [m]
  (let [[x y] (mat/shape m)]
    (->Matrix (.ge (.-me m) (DoubleMatrix/rand x y))
              {})))

(defn probs-hs-given-vs [[^Matrix weights ^Matrix h-bias] v-states]
  (cond-prob-batch weights h-bias v-states))

(defn probs-vs-given-hs [[^Matrix weights ^Matrix v-bias] h-states]
  (cond-prob-batch (->Matrix (.transpose ^DoubleMatrix (.-me weights))
                             {})
                   v-bias h-states))

(defn cd-batch
  "Implements contrastive divergence with duration steps (CD-k),
  a duration (k) = 1 is often used and performs reasonably."
  [[w vbs hbs] v-data h-data duration]
  (let [v-count (count v-data)
        h-count (count h-data)
        init-chain [v-data h-data]]
    (reduce
     (fn [chain i]
       (let [last (get chain (dec (count chain)))
             v-recons (->> last
                           sample-binary-matrix
                           (probs-vs-given-hs [w vbs]))]
         (-> chain
             (conj v-recons)
             (conj (->> v-recons
                        sample-binary-matrix
                        (probs-hs-given-vs [w hbs]) )))))
     init-chain
     (range duration))))


(defn calc-batch-up
  "Calculate the total updates on the model (usually calculated through a cd chain,
  approximating <v-data,h-data> - <v-model,h-model>. "
  [v-model-batch h-model-batch v-data-batch h-data-batch]
  [(sub (reduce add (map outer-product h-data-batch v-data-batch))
        (reduce add (map outer-product h-model-batch v-model-batch)))
   (sub (reduce add v-data-batch)
        (reduce add v-model-batch))
   (sub (reduce add h-data-batch)
        (reduce add h-model-batch))])

(defn train-cd-batch [[w vbs hbs] batches rate k]
  (reduce (fn [[w vbs hbs] v-probs-batch]
            (let [v-data-batch (sample-binary-matrix v-probs-batch)
                  h-probs-batch (probs-hs-given-vs [w hbs] v-data-batch)
                  chain (cd-batch [w vbs hbs] v-probs-batch h-probs-batch k)
                  ;; fix copying https://github.com/tel/clatrix/issues/36#issuecomment-66277605
                  up (calc-batch-up (map (comp mat/matrix vector)
                                         (rows (get chain (- (count chain) 2))))
                                    (map (comp mat/matrix vector)
                                         (rows (get chain (dec (count chain)))))
                                    (map (comp mat/matrix vector)
                                         (rows v-probs-batch))
                                    (map (comp mat/matrix vector)
                                         (rows h-probs-batch)))]
              (map #(add %1 (mul %2 (/ rate (count v-probs-batch))))
                   [w vbs hbs]
                   up)))
          [w vbs hbs]
          batches))


(defrecord JBlasRBM [^Matrix restricted-weights ^Matrix v-biases ^Matrix h-biases]
  PBoltzmannMachine
  (-biases [this] (vec (concat v-biases h-biases)))
  (-weights [this] (full-matrix restricted-weights))


  PRestrictedBoltzmannMachine
  (-v-biases [this] v-biases)
  (-h-biases [this] h-biases)
  (-restricted-weights [this] restricted-weights)

  PContrastiveDivergence
  (-train-cd [this batches epochs learning-rate k]
    (let [[weights v-bias h-bias]
          (reduce (fn [model step]
                    (println "Training epoch" step "rate:" (/ learning-rate step))
                    (train-cd-batch model
                                    batches
                                    (/ learning-rate step)
                                    k))
                  [restricted-weights v-biases h-biases]
                  (range 1 (inc epochs)))]
      (assoc this :restricted-weights weights :v-biases v-bias :h-biases h-bias
             :trained {:epochs epochs
                       :learning-rate learning-rate
                       :batch-size (count (first batches))
                       :cd-steps k})))


  PSample
  (-sample-gibbs [this iterations start-state particles]
    (let [w (-weights this)
          bs (-biases this)]
      (reduce (fn [chain step]
                (let [last (get chain (dec (count chain)))]
                  (conj chain (->> (range (count bs))
                                   (map (partial cond-prob w bs last))
                                   sample-binary))))
              [(vec start-state)]
              (range iterations)))))


(defmethod print-method JBlasRBM [v ^java.io.Writer w]
  (.write w (str "#boltzmann.jblas/JBlasRBM "
                 (merge (into {} v)
                        {:restricted-weights
                         (mapv (comp vec seq) (:restricted-weights v))
                         :v-biases (vec (seq (:v-biases v)))
                         :h-biases (vec (seq (:h-biases v)))}))))


(defn create-jblas-rbm
  ([v-count h-count]
     (->JBlasRBM
      (matrix (repeatedly h-count
                          (fn [] (sample-normal v-count :mean 0 :sd 0.01))))
      (matrix [(vec (sample-normal v-count :mean 0 :sd 0.01))])
      (matrix [(vec (repeat h-count 0))])))
  ([restricted-weights v-biases h-biases]
     (->JBlasRBM (matrix restricted-weights)
                 (matrix [v-biases])
                 (matrix [h-biases]))))

(defn load-jblas-rbm [{:keys [restricted-weights v-biases h-biases] :as rbm}]
  (merge (->JBlasRBM (matrix restricted-weights)
                     (matrix [v-biases])
                     (matrix [h-biases]))
         (dissoc rbm :restricted-weights :v-biases :h-biases)))


(defn bin-label [l]
    (assoc (vec (repeat 10 0)) (int l) 1))

(defn max-index [probs]
    (let [m (apply max probs)]
      (reduce (fn [sum p] (if (= m p) (reduced sum)
                             (inc sum)))
              0
              probs)))

(defn classification-rate [labels classified]
    (/ (count (filter (partial apply =)
                      (partition 2 (interleave labels classified))))
       (count classified)))

(comment
  ;; require some more stuff for live coding, should not be in the library
  (require '[boltzmann.mnist :as mnist]
           '[criterium.core :refer [bench]])
  (require '[boltzmann.mnist :as mnist]
           '[boltzmann.core :refer :all]
           '[boltzmann.jblas :refer [create-jblas-rbm]]
           '[clojure.core.matrix :as mat])
  (mat/current-implementation)
  ;(mat/set-current-implementation :persistent-vector)

  (future (def images (mnist/read-images "resources/train-images-idx3-ubyte"))
          (def batches (doall (map matrix (partition 10 images)))))

  ;; investigate theano error fn

  (def labels (mnist/read-labels "resources/train-labels-idx1-ubyte"))

  (def labeled-images
    (map #(float-array (concat %1 %2)) (map bin-label labels) images))

  (def labeled-batches
    (doall (map matrix (partition 100 labeled-images))))

  (require '[clojure.edn :as edn])

  (def trained
    (edn/read-string {:readers {'boltzmann.jblas/JBlasRBM load-jblas-rbm}}
                     (slurp "resources/mnist-labeled.edn")))

  (spit "resources/mnist-labeled.edn" (prn-str trained))



  (def trained
    (let [rbm (create-jblas-rbm 794 100)]
      (time (-train-cd rbm labeled-batches 4 0.01 1))))

  (->> images
       (take 100)
       (map (comp matrix vector))
       (map #(->> %
                  (probs-hs-given-vs [(:restricted-weights trained)
                                      (:h-biases trained)])
                  (probs-vs-given-hs [(:restricted-weights trained)
                                      (:v-biases trained)])))
       (map #(partition 28 %))
       (mnist/tile 10)
       mnist/render-grayscale-float-matrix
       mnist/view)

  (def test-images (map (comp float-array concat) (repeat (repeat 10 0)) images))



  (def classified
    (->> test-images
         (take 6000)
         (map (comp matrix vector))
         (map #(->> %
                    (probs-hs-given-vs [(:restricted-weights trained)
                                        (:h-biases trained)])
                    (probs-vs-given-hs [(:restricted-weights trained)
                                        (:v-biases trained)])))
         (map (partial take 10))
         (map max-index)))



  (float (classification-rate labels classified))

  (->> test-images
       (take 100)
       (map (comp matrix vector))
       (map #(->> %
                  (probs-hs-given-vs [(:restricted-weights trained)
                                      (:h-biases trained)])
                  (probs-vs-given-hs [(:restricted-weights trained)
                                      (:v-biases trained)])))
       (map (partial drop 10))
       (map #(partition 28 %))
       (mnist/tile 10)
       mnist/render-grayscale-float-matrix
       mnist/view)
  )
