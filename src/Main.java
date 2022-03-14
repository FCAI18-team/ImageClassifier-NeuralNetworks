import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        //Load Data
        File[] images = new File("Cats & Dogs Sample Dataset").listFiles();
        assert images != null;
        ImageData[] data = new ImageData[images.length];
        for (int i = 0; i < images.length; i++) {
            data[i] = new ImageData();
            data[i].setPixels(ImageHandler.ImageToIntArray(images[i]));
            data[i].setLabel(images[i].getName().contains("cat") ? 0 : 1); // 0 is a cat , 1 is a dog
        }

        //Shuffle
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);

        //Split the data into training (75%) and testing (25%) sets
        int[][] trainingSetFeatures, testingSetFeatures;
        int[] trainingSetLabels, testingSetLabels;

        int trainingSetSize = (int) (data.length * 0.75);
        trainingSetFeatures = new int[trainingSetSize][];
        trainingSetLabels = new int[trainingSetSize];

        for (int i = 0; i < trainingSetSize; i++) {
            trainingSetFeatures[i] = new int[data[i].pixels.length];
            System.arraycopy(data[i].pixels, 0, trainingSetFeatures[i], 0, data[i].pixels.length);
            trainingSetLabels[i] = data[i].label;
        }

        testingSetFeatures = new int[data.length - trainingSetSize][];
        testingSetLabels = new int[data.length - trainingSetSize];

        for (int i = 0; i < data.length - trainingSetSize; i++) {
            testingSetFeatures[i] = new int[data[i].pixels.length];
            System.arraycopy(data[i].pixels, 0, testingSetFeatures[i], 0, data[i].pixels.length);
            testingSetLabels[i] = data[i].label;
        }

        //Create the NN
        NeuralNetwork nn = new NeuralNetwork();

        //Set the NN architecture
        nn.init(data[0].pixels.length);

        //Train the NN
        nn.train(trainingSetFeatures, trainingSetLabels);

        //Test the model
        int[] predictedLabels = nn.predict(testingSetFeatures);
        double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
        System.out.println("-Model testing accuracy is " + accuracy + "%");

        //Save the model (final weights)
        nn.save("model.txt");

        //Load the model and use it on an image
        NeuralNetwork nn2 = new NeuralNetwork();
        nn2 = nn2.load("model.txt");
        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("sample.jpg"));
        int samplePrediction = nn2.predict(sampleImgFeatures);
        ImageHandler.showImage("sample.jpg");
        //Print "Cat" or "Dog"
        System.out.println(samplePrediction == 0 ? "-The sample is a cat" : "-The sample is a dog");
    }
}