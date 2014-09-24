(ns boltzmann.mnist
  (:require  [byte-streams :refer [to-byte-buffer]]
             [gloss.core :refer [defcodec compile-frame finite-frame
                                 ordered-map repeated]]
             [gloss.io :refer [encode decode]]
             [clojure.java.io :refer [input-stream]]))


(defn read-labels [file]
  (let [labels (compile-frame
                (ordered-map :magic :int32,
                             :labels (finite-frame :int32
                                                   (repeated :byte :prefix :none))))]
    (->> file
         input-stream
         to-byte-buffer
         (decode labels)
         :labels)))

(defcodec raw-image (repeat 28 (repeat 28 :ubyte)))

(defn read-images [file]
  (let [buffer (to-byte-buffer (input-stream file))
        images-frame (compile-frame
                      (ordered-map :magic :int32,
                                   :image-count :int32,
                                   :rows :int32,
                                   :columns :int32,
                                   :images (repeated raw-image :prefix :none)))
        raw-images (decode images-frame buffer)]
    (doall (mapv
            (fn [img] (mapv
                      (fn [row]
                        ;; float-array instead of pvec keeps us below
                        ;; 2 gig heap atm., take care with "dot"
                        ;; dot-product TODO
                        (float-array (mapv (fn [pixel]
                                             (float (/ (float pixel) 256)))
                                           row)))
                      img))
            (:images raw-images)))))



(defn view [image]
  (doto (javax.swing.JFrame. "MNIST Digit")
    (.add (proxy [javax.swing.JPanel] []
            (paintComponent [g]
              (proxy-super paintComponent g)
              (.drawImage g image 0 0 this))))
    (.setSize (.getWidth image) (.getHeight image))
    (.setResizable false)
    (.setVisible true)))


(defn draw-image [image]
  (let [w (count (first image))
        h (count image)
        buffered-image (java.awt.image.BufferedImage. w h java.awt.image.BufferedImage/TYPE_BYTE_GRAY)
        coords (for [x (range w) y (range h)] [x y (get-in image [y x])])]
    (doseq [[x y c] coords]
      (.setRGB buffered-image x y
               (.getRGB (java.awt.Color. (float c) (float c) (float c)))))
    buffered-image))



(comment
  (def images (read-images "resources/train-images-idx3-ubyte"))

  (-> (images 1)
      draw-image
      view)

  (def labels (read-labels "resources/train-labels-idx1-ubyte")))
