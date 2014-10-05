(ns boltzmann.core-test
  (:require [clojure.test :refer :all]
            [boltzmann.core :refer :all]
            [boltzmann.formulas :refer :all]
            [boltzmann.restricted :refer :all]))

(deftest prepare-batch-test
  (testing "Prepending of ones to batch for bias."
    (is (= (prepare-batch [[1 2] [3 4]])
           [[1 1 2] [1 3 4]]))))

(deftest boltz-cond-prob-batch-test
  (testing "Testing conditional boltzmann probability batch function."
    (is (= (boltz-cond-prob-batch [[0.1 0.2 0.3]
                                   [0.2 0.4 0.6]]
                                  [[1 0 1]
                                   [1 1 1]])
           '((0.598687660112452 0.6899744811276125)
             (0.6456563062257954 0.7685247834990178))))))

(deftest boltz-sampling-step-batch
  (testing "Sampling hidden and then visible probabilities."
    (is (= (let [l (->Layer [[0.1 0.1 0.1]
                             [0.2 0.3 0.4]] [0.8 0.7 0.6] [-0.1 -0.2])]
             (->> [[0.1 0.2 0.3]]
                  prepare-batch
                  (probs-hs-given-vs l)
                  prepare-batch
                  (probs-vs-given-hs l)))
           '((0.7209140523056718 0.7107439863826716 0.7003573555645741))))))
