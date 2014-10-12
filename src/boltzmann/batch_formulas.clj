(ns boltzmann.batch-formulas
  "Optimized batch based training routines."
  (:require [clatrix.core :refer [->Matrix]]
            [clojure.core.matrix :refer [zero-vector zero-matrix] :as mat])
  (:import [org.jblas MatrixFunctions DoubleMatrix]
           [clatrix.core Matrix]))


(defn outer-product [^Matrix a ^Matrix b]
  "Outer product (between two vectors given as 1-dimensional matrices)."
  (let [ac (count a)
        bc (count b)]
    (->Matrix (.rankOneUpdate ^DoubleMatrix (.-me (zero-matrix ac bc))
                              ^DoubleMatrix (.-me a)
                              ^DoubleMatrix (.-me b))
              {})))

(defn boltz-cond-prob-batch
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
