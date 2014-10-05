(ns boltzmann.formulas
  (:require [clojure.core.matrix :refer [dot get-row exp log matrix transpose
                                         zero-matrix join-along]]))

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

(defn boltz-cond-prob-batch [w batch]
  (->> (dot w (transpose batch))
       transpose
       ;; TODO implement sigmoid as matrix op through exp
       (map #(map σ %))))

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
