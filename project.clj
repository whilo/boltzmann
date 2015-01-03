(defproject net.polyc0l0r/boltzmann "0.1.1-SNAPSHOT"
  :description "Boltzmann Machine related deep learning techniques."
  :url "http://github.com/ghubber/boltzmann"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 ;; for data loading, should go into separate dep for data
                 [byte-streams "0.1.13"]
                 [clatrix "0.4.0"]
                 [gloss "0.2.2"]
                 [net.mikera/core.matrix "0.32.1"]
                 [org.clojure/math.combinatorics "0.0.4"]
                 [bigml/sampling "3.0"]
                 ;; replace sample-normal and move to dev deps
                 [incanter "1.5.5"]]
  :profiles {:dev {:dependencies [[criterium "0.4.3"]]}}
  ;; deactivate tiered compilation to allow proper benchmarks with criterium
  :jvm-opts ^:replace [])
