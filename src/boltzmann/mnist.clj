(ns boltzmann.mnist
  (:require  [byte-streams :refer [to-byte-buffer]]
             [gloss.core :refer [defcodec compile-frame finite-frame
                                 ordered-map repeated]]
             [gloss.io :refer [encode decode]]
             [clojure.java.io :refer [input-stream file]])
  (:import [javax.imageio ImageIO]))

;; for gzip have a look at https://gist.github.com/bpsm/1858654

(defn read-labels
  "Reads a label file for MNIST digits and returns a list of integers."
  [file]
  (let [labels (compile-frame
                (ordered-map :magic :int32,
                             :labels (finite-frame :int32
                                                   (repeated :byte :prefix :none))))]
    (->> file
         input-stream
         to-byte-buffer
         (decode labels)
         :labels)))

(defcodec raw-image (repeat (* 28 28) :ubyte))

(defn read-images
  "Reads a file of MNIST digits and returns a list of float-arrays for each image,
  containing gray-scale pixel values between 0 and 1."
  [file]
  (let [buffer (to-byte-buffer (input-stream file))
        images-frame (compile-frame
                      (ordered-map :magic :int32,
                                   :image-count :int32,
                                   :rows :int32,
                                   :columns :int32,
                                   :images (repeated raw-image :prefix :none)))
        raw-images (decode images-frame buffer)]
    (doall (mapv (fn [img] (float-array (map #(float (/ (float %) 256)) img)))
                 (:images raw-images)))))


(defn horizontal-tile [float-matrices] ;; TODO generalize to vertical
  (->> float-matrices
       (apply interleave)
       (apply concat)
       (partition (* (count float-matrices)
                     (count (first float-matrices))))))


(defn view [image]
  (doto (javax.swing.JFrame. "MNIST Digit")
    (.add (proxy [javax.swing.JPanel] []
            (paintComponent [g]
              (proxy-super paintComponent g)
              (.drawImage g image 0 0 this))))
    (.setSize (.getWidth image) (.getHeight image))
    (.setResizable false)
    (.setVisible true)))


(defn render-grayscale-float-matrix [m]
  (let [w (count (first m))
        h (count m)
        buffered-image (java.awt.image.BufferedImage.
                        w h java.awt.image.BufferedImage/TYPE_BYTE_GRAY)
        coords (for [x (range w) y (range h)] [x y (-> m (nth y) (nth x))])]
    (doseq [[x y c] coords]
      (.setRGB buffered-image x y
               (.getRGB (java.awt.Color. (float c) (float c) (float c)))))
    buffered-image))



(comment
  (def images (read-images "resources/train-images-idx3-ubyte"))

  (def rendered (->> (take 30 images)
                     (map #(partition 28 %))
                     horizontal-tile
                     render-grayscale-float-matrix))

  (view rendered)

  (ImageIO/write rendered "png" (file "/tmp/mnist.png"))

  (def labels (read-labels "resources/train-labels-idx1-ubyte")))
