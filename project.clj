(defproject boltzmann "0.1.0-SNAPSHOT"
  :description "Boltzmann Machine related deep learning techniques."
  :url "http://github.com/ghubber/boltzmann"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [;; for data loading, should go into separate dep for data
                 [byte-streams "0.1.13"]
                 [clatrix "0.4.0"]
                 [criterium "0.4.3"]
                 [gloss "0.2.2"]
                 ;; TODO move into dev profile again for release
                 [incanter "1.5.5"]
                 [net.mikera/core.matrix "0.29.1"]
                 [org.clojure/clojure "1.7.0-alpha2"]
                 [org.clojure/java.jdbc "0.3.5"]
                 [org.clojure/math.combinatorics "0.0.4"]]
  ;; deactivate tiered compilation to allow proper benchmarks with criterium
  :jvm-opts ^:replace [])
