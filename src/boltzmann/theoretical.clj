(ns boltzmann.theoretical
  (:require [boltzmann.protocols :refer :all]
            [boltzmann.matrix :refer [full-matrix]]
            [boltzmann.sample :refer [sample-binary]]
            [boltzmann.formulas :refer [cond-prob]]
            [clojure.core.matrix :refer [exp log add sub mul matrix
                                         transpose columns get-row outer-product] :as mat]
            [incanter.stats :refer [sample-normal]]))

(mat/set-current-implementation :persistent-vector)

(defn probs-h-given-v [[weights v-bias h-bias] v-state]
  (mapv (fn [i] (cond-prob weights h-bias v-state i))
        (range (count h-bias))))

(defn probs-v-given-h [[weights v-bias h-bias] h-state]
  (mapv (fn [i] (cond-prob (transpose weights) v-bias h-state i))
        (range (count v-bias))))

(defn cd
  "Implements contrastive divergence with duration steps (CD-k),
  a duration (k) = 1 is often used and performs reasonably."
  [[weights v-bias h-bias] v-data h-data duration]
  (let [v-count (count v-bias)
        h-count (count h-bias)
        init-chain [v-data h-data]]
    (reduce
     (fn [chain i]
       (let [v-recon (probs-v-given-h [weights v-bias h-bias]
                                      (sample-binary (get chain (dec (count chain)))))]
         (-> chain
             (conj v-recon)
             (conj (probs-h-given-v [weights v-bias h-bias]
                                    (sample-binary v-recon))))))
     init-chain
     (range duration))))

(defn calc-up
  "Calculate the total updates on the model (usually calculated through a cd chain,
  approximating <v-data,h-data> - <v-model,h-model>. "
  [v-model h-model v-data h-data]
  [(sub (outer-product h-data v-data)
        (outer-product h-model v-model))
   (sub v-data v-model)
   (sub h-data h-model)])

(defn train-cd-epoch [[weights v-bias h-bias] samples rate k]
  (reduce (fn [[weights v-bias h-bias] v-probs]
            (let [v-data (sample-binary v-probs)
                  h-probs (probs-h-given-v [weights v-bias h-bias] v-data)
                  h-data (sample-binary h-probs)
                  chain (cd [weights v-bias h-bias] v-data h-data k)
                  up (calc-up (get chain (- (count chain) 2))
                              (get chain (dec (count chain)))
                              v-probs h-probs)]
              (map #(add %1 (mul %2 rate))
                   [weights v-bias h-bias]
                   up)))
          [weights v-bias h-bias]
          samples))

(defn- unpack-batches [batches]
  (let [datum (ffirst batches)]
    (if (or (seq? datum)
            (vector? datum))
      (do (println "Minibatch learning is an optimization and not implemented for the theoretical RBM implementation. Unpacking batches and training on singular samples.")
          (apply concat batches))
      batches)))


(defrecord TheoreticalRBM [restricted-weights v-biases h-biases]
  PBoltzmannMachine
  (-biases [this] (vec (concat v-biases h-biases)))
  (-weights [this] (full-matrix restricted-weights))

  PRestrictedBoltzmannMachine
  (-v-biases [this] h-biases)
  (-h-biases [this] v-biases)
  (-restricted-weights [this] restricted-weights)

  PContrastiveDivergence
  (-train-cd [this batches epochs learning-rate k]
    (let [samples (unpack-batches batches)
          [weights v-bias h-bias]
          (reduce (fn [model step]
                    (train-cd-epoch model samples
                                    (/ learning-rate step)
                                    k))
                  [restricted-weights v-biases h-biases]
                  (range 1 (inc epochs)))]
      (assoc this :restricted-weights weights :v-biases v-bias :h-biases h-bias)))

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

(defn create-theoretical-rbm
  ([v-count h-count]
     (let [scaled-sd (/ 0.01 (+ v-count h-count))]
       (->TheoreticalRBM
        (repeatedly h-count
                    (fn [] (sample-normal v-count :mean 0 :sd scaled-sd)))
        (vec (sample-normal v-count :mean 0 :sd scaled-sd))
        (vec (repeat h-count 0)))))
  ([restricted-weights v-biases h-biases]
     (->TheoreticalRBM restricted-weights v-biases h-biases)))
