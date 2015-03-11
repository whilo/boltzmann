(ns boltzmann.formulas
  (:require [clojure.core.matrix :refer [add dot mul mget get-row exp log matrix transpose] :as mat]
            [clojure.math.combinatorics :refer [cartesian-product]]
            [boltzmann.protocols :refer [PRestrictedBoltzmannMachine -weights -biases
                                         -v-biases -h-biases -restricted-weights]]
            [bigml.sampling.random :as rnd]))

(def sum (partial reduce + 0))

(def prod (partial reduce * 1))

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
  ([bm z]
   (energy (-weights bm) (-biases bm)))
  ([weights bias z]
   (+ (- (/ (dot z (dot z weights))
            2))
      (- (dot bias z)))))

(defn state-space
  "Generates binary state space of boltzmann machine for num units."
  [num]
  (mapv vec (apply cartesian-product (repeat num [0 1]))))


(defn boltz-partition
  "Partition function for normalization of Boltzmann probability."
  ([bm]
   (if (not (extends? PRestrictedBoltzmannMachine  (type bm)))
     (boltz-partition (-weights bm) (-biases bm))
     (let [vb (-v-biases bm)
           hb (-h-biases bm)
           w (-restricted-weights bm)
           vc (count vb)
           hc (count hb)
           [sb lb w] (if (> vc hc) [hb vb (transpose w)] [vb hb w])
           sc (count sb)
           lc (count lb)
           _ (when (> sc 20)
               (throw (ex-info "This state space is intractable to calculate
                      the partition function exactly."
                               {:type :partition-fn-intractable
                                :state-space-size sc
                                :bm bm})))]

       (time (sum (map (fn [z]
                         (* (exp (dot z sb))
                            (prod (map (fn [j]
                                         (+ 1 (exp (+ (lb j)
                                                      (sum (map (fn [i] (* (z i) ((w i) j)))
                                                                (range sc)))))))
                                       (range lc)))))
                       (state-space sc)))))))
  ([weights bias]
   (when (> (count bias) 20)
     (throw (ex-info "This state space is intractable to calculate
                      the partition function exactly."
                     {:type :partition-fn-intractable
                      :state-space-size (count bias)
                      :weights weights
                      :bias bias})))
   (boltz-partition weights bias (state-space (count bias))))
  ([weights bias states]
   (time (reduce (fn [sum z]
                   (+ sum (Math/exp (- (energy weights bias z)))))
                 0
                 states))))

(def boltz-partition-mem (memoize boltz-partition))

(comment
  (def trbm (create-theoretical-rbm 100 18))
  (boltz-partition-mem trbm))


(defn prob
  "Calculates the theoretical probability of the Boltzmann machine
  of state x given states xs."
  ([bm z]
   (let [Z (boltz-partition-mem bm)
         E (energy bm)]
     (* (/ 1
           Z) (Math/exp (- E)))))
  ([weights bias z]
   (let [Z (boltz-partition-mem weights bias)
         E (energy weights bias z)]
     (* (/ 1
           Z) (Math/exp (- E))))))



(defn sample-probs [samples]
  (let [c (count samples)]
    (->> (frequencies samples)
         (map (fn [[k v]] [k (float (/ v c))]))
         (into {}))))

(defn dkl
  "Kullback-Leibler divergence between a Boltzmann distribution and an
  empirical distribution."
  [weights bias states]
  (let [v-count (count (first states))
        q (sample-probs states)
        p (reduce (fn [f s] (update-in f [(take v-count s)]
                             (fnil + 0)
                             (prob weights bias s)))
                     {}
                     (state-space (count bias)))
        sp (state-space v-count)]
    (reduce (fn [sum x]
              (+ sum (* (p x) (Math/log (/ (p x)
                                           (q x))))))
            0
            sp)))

(defn free-entropy [bm]
    (- (log (boltz-partition bm))))
