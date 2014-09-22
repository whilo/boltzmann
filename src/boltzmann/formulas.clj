(ns boltzmann.formulas
  (:require [clojure.core.matrix :refer [dot get-row exp log matrix]]))


(defn σ [x]
  (/ 1
     (+ 1 (exp (- x)))))

(defn boltz-energy [weights bias x]
  (+ (- (/ (dot x (dot x weights))
           2))
     (- (dot bias x))))

(defn boltz-prob [weights bias xs x]
  (let [Z (reduce (fn [sum x]
                    (+ sum (exp (- (boltz-energy weights bias x)))))
                  0
                  xs)]
    (* (/ 1
          Z) (exp (- (boltz-energy weights bias x))))))

(defn boltz-cond-prob [w bs x i]
  (σ (+ (dot x (get-row w i)) (get bs i))))

(defn full-matrix
  "Expands a matrix from a restricted Boltzmann machine, which only
  covers visible and hidden units, into a symmetric matrix with zero
  blocks between units of each layer. This is used to use the normal
  gibbs-sampler to sample from the Boltzmann machine."
  [visi-hidden-matrix]
  ;; could be shorter, more declarative; still straight port from python
  (let [v-count (count (first visi-hidden-matrix))
        h-count (count visi-hidden-matrix)
        total (+ v-count h-count)
        M (matrix (repeat total (repeat total 0)))]
    (->> (for [i (range h-count)
               j (range v-count)]
           [i j])
         (reduce (fn [M [i j]]
                   (-> M
                       (assoc-in [(+ v-count i) j] (get-in visi-hidden-matrix [i j]))
                       (assoc-in [(- (dec v-count) j) (+ v-count i)]
                                 (get-in visi-hidden-matrix [i (- (dec v-count) j)]))))
                 M))))
