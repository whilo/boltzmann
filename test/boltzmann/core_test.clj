(ns boltzmann.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :refer [matrix] :as mat]
            [boltzmann.layer :refer :all]
            [boltzmann.formulas :refer :all]
            [boltzmann.batch-formulas :refer :all]
            [boltzmann.batch-train :refer :all]))

(mat/set-current-implementation :clatrix)

(deftest boltz-cond-prob-batch-test
  (testing "Testing conditional boltzmann probability batch function."
    (is (= (boltz-cond-prob-batch (matrix [[0.1 0.2 0.3]
                                           [0.2 0.4 0.6]])
                                  (matrix [[0.0 0.2]])
                                  (matrix [[1 0 1]
                                           [1 1 1]]))
           '((0.598687660112452 0.7310585786300049)
             (0.6456563062257954 0.8021838885585818))))))

(deftest boltz-sampling-step-batch
  (testing "Sampling hidden and then visible probabilities."
    (is (= (let [l (create-layer [[0.1 0.1 0.1]
                                  [0.2 0.3 0.4]]
                                 [[0.8 0.7 0.6]]
                                 [[-0.1 -0.2]])]
             (->> (matrix [[0.1 0.2 0.3]])
                  (probs-hs-given-vs l)
                  (probs-vs-given-hs l)))
           '((0.7209140523056718 0.7107439863826716 0.7003573555645741))))))


(deftest outer-product-test
  (testing "JBlas optimized outer-product."
    (is (= (outer-product
            (mat/matrix [(range 1 3)])
            (mat/matrix [(range 1 4)]))
           '((1.0 2.0 3.0)
             (2.0 4.0 6.0))))))
