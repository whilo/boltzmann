(ns boltzmann.visualize
  (:import [javax.imageio ImageIO]))


(defn horizontal-tile [float-matrices]
  (->> float-matrices
       (apply interleave)
       (apply concat)
       (partition (* (count float-matrices)
                     (count (first float-matrices))))))

(defn tile [width float-matrices]
  (->> float-matrices
       (partition width)
       (map horizontal-tile)
       (apply concat)))

(defn view [image & {:keys [title]
                     :or {title "MNIST Digits"}}]
  (doto (javax.swing.JFrame. title)
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

(defn receptive-fields [width x]
  (mapv #(mapv vec (partition width %)) x))

(comment
  (def rendered (->> (take 100 images)
                     (map #(partition 28 %))
                     (tile 10)
                     render-grayscale-float-matrix))

  (view rendered)

  (ImageIO/write rendered "png" (file "/tmp/mnist.png")))
