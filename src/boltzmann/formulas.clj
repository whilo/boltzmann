(ns boltzmann.formulas
  (:require [clojure.core.matrix :refer [add dot mul mget get-row exp log matrix transpose] :as mat]
            [clojure.math.combinatorics :refer [cartesian-product]]
            [bigml.sampling.random :as rnd]))

(defn create-seeded-rand [seed]
  (let [prng (rnd/create :seed seed)]
    (fn [] (rnd/next-double! prng))))

(defn σ "Sigmoid function." [x]
  (/ 1
     (+ 1 (exp (- x)))))

(defn cond-prob
  "Conditional probability for w(eights) and b(ia)s given state x in unit i."
  [w bs x i]
  (σ (+ (dot x (get-row w i)) (mget bs i))))

(defn σ' [x]
  (* (σ x) (σ (- x))))

(defn gauss [mean std x]
  (* (/ 1
        (* (Math/sqrt (* 2 Math/PI)) std))
     (exp (- (/ (Math/pow (- mean x) 2)
                (* 2 std std))))))

(defn softplus [x]
  (log (+ 1 (exp x))))

(defn energy
  "Calculates the energy of a state configuration x in a Boltzmann machine."
  [weights bias z]
  (+ (- (/ (dot z (dot z weights))
           2))
     (- (dot bias z))))

(defn state-space
  "Generates binary state space of boltzmann machine for num units."
  [num]
  (mapv vec (apply cartesian-product (repeat num [0 1]))))


(defn boltz-partition
  "Partition function for normalization of Boltzmann probability."
  ([weights bias]
     (boltz-partition weights bias (state-space (count bias))))
  ([weights bias states]
     (reduce (fn [sum z]
               (+ sum (Math/exp (- (energy weights bias z)))))
             0
             states)))

(defn prob
  "Calculates the theoretical probability of the Boltzmann machine
  of state x given states xs."
  [weights bias z]
  (let [Z (boltz-partition weights bias)
        E (energy weights bias z)]
    (* (/ 1
          Z) (Math/exp (- E)))))


(defn dkl
  "Kullback-Leibler divergence between a Boltzmann distribution and an
  empirical distribution."
  [weights bias states]
  (let [v-count (count (first states))
        q (a/sample-freqs states)
        p (reduce (fn [f s] (update-in f [(take v-count s)]
                             (fnil + 0)
                             (f/prob weights bias s)))
                     {}
                     (f/state-space (count bias)))
        sp (f/state-space v-count)]
    (reduce (fn [sum x]
              (+ sum (* (p x) (Math/log (/ (p x)
                                           (q x))))))
            0
            sp)))

(defn free-entropy [bm]
    (- (log (boltz-partition bm))))
