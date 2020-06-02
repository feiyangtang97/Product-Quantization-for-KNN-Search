/**
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Maaike Van Roy, 2020
 */

import java.util.*;


/**
 * To use the provided KMeans implementation, add the jar to the build path, 
 * import like this and use like any other class.
 */
import assignment.KMeans; 


public class PQkNN {

    private int n; // Amount of subvectors
    private int k; // k of k-Means
    private int[] trainlabels;
    private int[][] compressedData;
    private ArrayList<double[][]> subvectorCentroids;
    private double[][] distances;

    /**
     * save distance and label when k-nearest neighbor searches
     */
    class DistanceLabel implements Comparable<DistanceLabel> {
        double dis;
        int label;

        public DistanceLabel(double dis, int label) {
            this.dis = dis;
            this.label = label;
        }

        @Override
        public int compareTo(DistanceLabel other) {
            return Double.compare(this.dis, other.dis);
        }
    }

    /**
     * Contruct a new instance of kNN with Product Quantization
     *
     * @param n Amount of subvectors
     * @param c Determines the amount of clusters for KMeans, i.e., k = 2**c
     */
    public PQkNN(int n, int c) {

        this.n = n;
        this.k = (int) Math.pow(2, c);
    }

    /**
     * Split a vector into several subvectors
     *
     * @param sampleVector The given out-of-sample 1D image.
     * @return k sub vectors
     */
    // Split one 1D vector into multiple subvectors, # of subvectors are determined by n
    private int[][] split2SubVectors(int[] sampleVector) {
        int sampleLen = sampleVector.length;
        int subVecSize = (int) Math.ceil(sampleLen * 1.0 / n);
        int[][] subVectors = new int[n][subVecSize];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < subVecSize; j++) {
                if (i * subVecSize + j < sampleLen) {
                    subVectors[i][j] = sampleVector[i * subVecSize + j];
                }
            }
        }
        return subVectors;
    }

    /**
     * Dividing each row of sample vectors in training data into several sub vectors
     *
     * @param traindata train data
     * @return
     */
    // Split training data into multiple subvectors, will cluster each subvectors later to get a clutser centre
    private int[][][] getSubVectors(int[][] traindata) {
        int subVecSize = (int) Math.ceil(traindata[0].length * 1.0 / k);
        // 3d vector, the 2D vector before were divided into n parts
        // traindata.length is the # of training data
        // subVecSize is # of features in each divided training data
        int[][][] traindataSplited = new int[n][traindata.length][subVecSize];
        // process each sample
        for (int sampleIdx = 0; sampleIdx < traindata.length; sampleIdx++) {
            int[] sample = traindata[sampleIdx];
            // 1D vector split into multiple 1D vector
            int[][] subVectors = split2SubVectors(sample);
            for (int subIdx = 0; subIdx < subVectors.length; subIdx++) {
                int[] subVec = subVectors[subIdx];
                traindataSplited[subIdx][sampleIdx] = subVec;
            }
        }
        return traindataSplited;
    }

    /**
     * Compress the given training examples via the product quantization method (see assignment for paper and blog post).
     * The necessary data structures are to be stored within class instantiation.
     *
     * @param traindata   The training examples, 2D integer matrix where each row represents an image.
     * @param trainlabels Labels for the training examples, 0..9
     */
    public void compress(int[][] traindata, int[] trainlabels) {
        this.trainlabels = trainlabels; // You will need these during prediction.
        this.compressedData = new int[traindata.length][this.n];
        this.subvectorCentroids = new ArrayList<>();

        int[][][] traindataSplited = getSubVectors(traindata);
        // K-means clustering is used for each set of sub vectors to get n cluster centers
        // 2d vector splitted into multiple 2d vector, each one represents one part of data
        // now cluster each 2d vector, get cluster centre
        for (int i = 0; i < traindataSplited.length; i++) {
            int[][] subVectors = traindataSplited[i];
            System.out.printf("Total amount %d, Now cluster No. %d\n", traindataSplited.length, i);
            KMeans km = new KMeans(traindata.length, this.k);
            subvectorCentroids.add(km.fit(subVectors));
        }
        // Generate compressed features
        // original 1D vectors were splitted into multiple, calculate clustering centre for each as a feature
        // this is how the compression is done
        for (int sampleIdx = 0; sampleIdx < traindata.length; sampleIdx++) {
            for (int compressFeaIdx = 0; compressFeaIdx < n; compressFeaIdx++) {
                KMeans km = new KMeans(traindata.length, this.k);
                int code = km.predict(traindataSplited[compressFeaIdx][sampleIdx], subvectorCentroids.get(compressFeaIdx));
                compressedData[sampleIdx][compressFeaIdx] = code;
            }
        }
    }

    /**
     * Predicts the label of a given 1D-image example based on the PQkNN algorithm.
     *
     * @param testsample       The given out-of-sample 1D image.
     * @param nearestNeighbors k in kNN
     * @return test image classification (0..9)
     */
    public int predict(int[] testsample, int nearestNeighbors) {
        this.distances = new double[this.k][this.n];

        // after subvectors obtained from splitting, calculate distance from each subvector to each clustering centre
        // k is # of clustering centre, aka the feature's range, n is # of features
        int[][] subVectors = split2SubVectors(testsample);
        for (int subIdx = 0; subIdx < n; subIdx++) {
            for (int centerIdx = 0; centerIdx < subvectorCentroids.get(subIdx).length; centerIdx++) {
                this.distances[centerIdx][subIdx] = calculateDistance(subVectors[subIdx], subvectorCentroids.get(subIdx)[centerIdx]);
            }
        }

        // we get distance from each subvector to each clustering centre from last step
        // now each subvector will belong to a clustering centre
        // Calculate the distance to each sample, and use the priority queue to take the label of top k distance
        Queue<DistanceLabel> distanceLabels = new PriorityQueue<>();
        for (int sampleIdx = 0; sampleIdx < compressedData.length; sampleIdx++) {
            double dis = 0;
            for (int feaIdx = 0; feaIdx < this.n; feaIdx++) {
                dis += distances[compressedData[sampleIdx][feaIdx]][feaIdx];
            }
            int label = trainlabels[sampleIdx];
            distanceLabels.add(new DistanceLabel(dis, label));
        }
        // Calculate the most frequent tags in the top k neighborhood
        // take top 100 out from distance rankings
        Map<Integer, Integer> labelCnt = new HashMap<>();
        for (int i = 0; i < nearestNeighbors; i++) {
            int label = distanceLabels.poll().label;
            labelCnt.putIfAbsent(label, 0);
            labelCnt.put(label, labelCnt.get(label) + 1);
        }

        int label = 0;
        int maxCnt = 0;
        for (Map.Entry<Integer, Integer> entry : labelCnt.entrySet()) {
            if (entry.getValue() > maxCnt) {
                maxCnt = entry.getValue();
                label = entry.getKey();
            }
        }
        return label;
    }

    /**
     * Calculate the distance between the given example and a centroid.
     *
     * @param example1 The given example.
     * @param example2 The given centroid.
     * @return The distance.
     */
    private static double calculateDistance(int[] example1, double[] example2) {
        double sum = 0.0;
        for (int i = 0; i < example1.length; ++i) {
            sum += Math.pow(example1[i] - example2[i], 2);
        }
        return sum;
    }
}

