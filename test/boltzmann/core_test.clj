(ns boltzmann.core-test
  (:refer-clojure :exclude [partition])
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :refer [matrix] :as mat]
            [boltzmann.theoretical :refer [create-theoretical-rbm]]
            [boltzmann.formulas :refer [cond-prob boltz-partition]]
            [boltzmann.matrix :refer [full-matrix]]
            [boltzmann.jblas :refer [cond-prob-batch create-jblas-rbm
                                     probs-hs-given-vs probs-vs-given-hs
                                     outer-product]]))

(mat/set-current-implementation :clatrix)

(deftest cond-prob-batch-test
  (testing "Testing conditional boltzmann probability batch function."
    (is (= (cond-prob-batch (matrix [[0.1 0.2 0.3]
                                     [0.2 0.4 0.6]])
                            (matrix [[0.0 0.2]])
                            (matrix [[1 0 1]
                                     [1 1 1]]))
           '((0.598687660112452 0.7310585786300049)
             (0.6456563062257954 0.8021838885585818))))))

(deftest sampling-step-batch
  (testing "Sampling hidden and then visible probabilities."
    (is (= (let [weights (matrix [[0.1 0.1 0.1]
                                  [0.2 0.3 0.4]])
                 v-b (matrix [[0.8 0.7 0.6]])
                 h-b (matrix [[-0.1 -0.2]])]
             (->> (matrix [[0.1 0.2 0.3]])
                  (probs-hs-given-vs [weights h-b])
                  (probs-vs-given-hs [weights v-b])))
           '((0.7209140523056718 0.7107439863826716 0.7003573555645741))))))


(deftest outer-product-test
  (testing "JBlas optimized outer-product."
    (is (= (outer-product
            (matrix [(range 1 3)])
            (matrix [(range 1 4)]))
           '((1.0 2.0 3.0)
             (2.0 4.0 6.0))))))




(deftest partition-fn-test
  (testing "Testing full vs. restricted partition function implementation."
    (is (let [w (vec (repeat 2 (vec (repeat 4 0))))
              vb (vec (repeat 4 1))
              hb (vec (repeat 2 2))
              trbm (create-theoretical-rbm w vb hb)
              restr (boltz-partition trbm)
              full (boltz-partition (full-matrix w) (concat vb hb))]
          (if (< (- restr full)
                 1e-3)
            true
            [(- restr full) restr full])))))
