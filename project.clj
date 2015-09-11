(defproject es.topiq/boltzmann "0.1.3-SNAPSHOT"
  :description "Boltzmann Machine related deep learning techniques."
  :url "http://github.com/whilo/boltzmann"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.clojure/core.async "0.1.346.0-17112a-alpha"]
                 ;; for data loading, should go into separate project for data
                 [byte-streams "0.2.0"]
                 [clatrix "0.5.0"]
                 [gloss "0.2.5"]
                 [net.mikera/core.matrix "0.40.0"]
                 [org.clojure/math.combinatorics "0.1.1"]
                 [bigml/sampling "3.0"]
                 ;; replace sample-normal and move to dev deps
                 [incanter "1.5.5"]]
  :profiles {:dev {:dependencies [[criterium "0.4.3"]]}}
  ;; deactivate tiered compilation to allow proper benchmarks with criterium
  :jvm-opts ^:replace [])
