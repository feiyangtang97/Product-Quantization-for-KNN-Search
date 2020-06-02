import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Thomas Dierckx, 2019
 * <p>
 * DO NOT ALTER ANYTHING
 */

public class DataReader {

    /**
     * Reads in training or test set (.csv)
     * It assumes the label is given in the first column in the csv file.
     * It also assumes ',' as delimiter, no header row and no white space characters.
     * One row denotes one example, where the columns (with exception of the first one) are the features.
     * <p>
     * Example usage:
     * DataSet data = readData('training.csv', 60000)
     * int[][] data = data.getData();
     * int [] labels = data.getLabels();
     * <p>
     * ( training set length for MNIST is 60000 images )
     * ( test set length for MNIST is 10000 images )
     *
     * @param filepath
     * @param length:  amount of examples (images in this case)
     * @return DataSet object (contains int[][] data and int[] labels)
     */
    public static DataSet readData(String filepath, int length) {

        int[][] data = new int[length][784];
        int[] labels = new int[length];
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            int imageNr = 0;
            while ((line = br.readLine()) != null && imageNr < length) {
                String[] values = line.split(",");
                for (int i = 0; i < values.length - 1; i++) // skip label
                    data[imageNr][i] = Integer.parseInt(values[i + 1]);

                labels[imageNr] = Integer.parseInt(values[0]);
                imageNr++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new DataSet(data, labels);
    }

    /**
     * Wrapper class for a dataset without labels, and its labels.
     */
    public static class DataSet {
        int[][] data; // data represents 1d images (each row = image)
        int[] labels; // labels corresponding to images in data

        private DataSet(int[][] data, int[] labels) {
            this.data = data;
            this.labels = labels;
        }

        /**
         * Get data without labels.
         *
         * @return int[][] trainingData
         */
        public int[][] getData() {
            return this.data;
        }

        /**
         * Get labels of data.
         *
         * @return int[] labels
         */
        public int[] getLabels() {
            return this.labels;
        }
    }

}
