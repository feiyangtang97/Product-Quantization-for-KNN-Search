import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;

public class Main {
    public static void main(String[] args) throws InterruptedException {
        long start = System.currentTimeMillis();
        PQkNN pQkNN = new PQkNN(30, 8);

        // Read data
        DataReader.DataSet trainData = DataReader.readData("./mnist_train.csv", 60000);


        // Compress data
        pQkNN.compress(trainData.data, trainData.labels);

        long end = System.currentTimeMillis();
        System.out.printf("Compressed data elapsed time %.2f seconds\n\n", (end - start) / 1000.0);

        // Recalculate prediction time consumption
        start = System.currentTimeMillis();
        DataReader.DataSet testData = DataReader.readData("./mnist_test.csv", 10000);

        System.gc();
        Thread.sleep(2000);
        MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        System.out.println("getHeapMemoryUsage " + memoryMXBean.getHeapMemoryUsage() + "\n");


        // total prediction #
        int total = 0;
        // correct prediction #
        int predRight = 0;
        for (int i = 0; i < testData.data.length; i++) {
            // make prediction
            int predLabel = pQkNN.predict(testData.data[i], 100);
            total += 1;
            if (predLabel == testData.labels[i]) {
                predRight += 1;
            }
        }
        end = System.currentTimeMillis();
        System.out.printf("Predict elapsed time %.2f seconds\n\n", (end - start) / 1000.0);
        System.out.printf("Accurancy: %.2f%%\n", predRight * 100.0 / total);

    }
}
