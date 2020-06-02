#####################
Remarks:
- Keep MNIST datasets under `./src/` the same with java and class files
- Python script is attached as well
#####################
Compile:
`cd /path/to/src`
`javac -cp KMeans.jar *.java`
#####################
Run Java experiments:
`java -cp KMeans.jar:src/*:. Main`
#####################
Run Python experiments:
`python main.py`
#####################
Output from Java:
Compressed data elapsed time 2724.48 seconds
getHeapMemoryUsage init = 134217728(131072K) used = 239529096(233915K) committed = 1040187392(1015808K) max = 1908932608(1864192K)
Predict elapsed time 47.95 seconds
Accurancy: 94.41%
#####################
Output from Python (sklearn):
Elapsed Time 734.29 seconds
Accurancy: 94.48%