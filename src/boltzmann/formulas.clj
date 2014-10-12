(ns boltzmann.formulas
  (:require [clojure.core.matrix :refer [add div dot mul mmul mget get-row exp log matrix transpose
                                         zero-matrix join-along] :as mat]))

(defn σ "Sigmoid function." [x]
  (/ 1
     (+ 1 (exp (- x)))))

(defn boltz-energy
  "Calculates the energy of a state configuration x in a Boltzmann machine."
  [weights bias x]
  (+ (- (/ (dot x (dot x weights))
           2))
     (- (dot bias x))))

(defn boltz-prob
  "Calculates the theoretical probability of the Boltzmann machine
  of state x given states xs."
  [weights bias xs x]
  (let [Z (reduce (fn [sum x]
                    (+ sum (exp (- (boltz-energy weights bias x)))))
                  0
                  xs)]
    (* (/ 1
          Z) (exp (- (boltz-energy weights bias x))))))

(defn boltz-cond-prob
  "Conditional probability for w(eights) and b(ia)s given state x in unit i."
  [w bs x i]
  (σ (+ (dot x (get-row w i)) (mget bs i))))

(defn sample-binary
  "Samples a vector of binary values from a vector of probabilities."
  [probabilities]
  (mapv (fn [p] (if (< (rand) p) 1 0)) probabilities))

(defn gibbs-sampler
  "Sample from a Boltzmann distribution given by w(eights) and b(ia)s."
  [w bs iterations]
  (reduce (fn [chain step]
            (let [last (get chain (dec (count chain)))]
              (conj chain (->> (range (count bs))
                               (map (partial boltz-cond-prob w bs last))
                               sample-binary))))
          [(repeat (count bs) 0)] ;; init
          (range iterations)))

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
