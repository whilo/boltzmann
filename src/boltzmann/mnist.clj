(ns boltzmann.mnist
  (:require  [byte-streams :refer :all]
             [gloss.core :refer :all]
             [gloss.io :refer [encode decode]]
             [clojure.java.io :as io]))


(defn read-labels [file]
  (let [labels (compile-frame
                (ordered-map :magic :int32,
                             :labels (finite-frame :int32
                                                   (repeated :byte :prefix :none))))]
    (decode labels (to-byte-array (io/input-stream file)))))


(comment
  (:magic (read-labels "resources/train-labels-idx1-ubyte"))
  ;; example from gloss wiki
  (def fr (compile-frame {:a :int16, :b :float32}))

  (encode fr {:a 1, :b 2})

  (decode fr *1))
