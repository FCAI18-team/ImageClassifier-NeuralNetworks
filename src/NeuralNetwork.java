import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class NeuralNetwork {
    private final int numOfNeurals = 50;
    private int numOfInputs;
    private final int numOfOutputs = 1;
    private double[][] weights;
    private double[][] innerWeights;
    private double[] outputValues = new double[numOfOutputs];
    double[] hiddenLayerValues = new double[numOfNeurals];
    private double[][] newInnerWeights;
    private final double n = 0.5;
    private final int epochs = 5000;

    NeuralModel neuralModel = new NeuralModel();

    public void initNeuralModel() {
        neuralModel.weights = new double[numOfNeurals][numOfInputs];
        neuralModel.innerWeights = new double[numOfOutputs][numOfNeurals];
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(weights[i], 0, neuralModel.weights[i], 0, weights[i].length);
        }
        for (int i = 0; i < innerWeights.length; i++) {
            System.arraycopy(innerWeights[i], 0, neuralModel.innerWeights[i], 0, innerWeights[i].length);
        }
    }

    public void init(int inputSize) {
        this.numOfInputs = inputSize;
        this.weights = new double[numOfNeurals][numOfInputs];
        this.innerWeights = new double[numOfOutputs][numOfNeurals];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                double r = Math.random();
                if (r > 0.5) {
                    weights[i][j] = Math.random();
                } else {
                    weights[i][j] = Math.random() * -1;
                }
            }
        }
        for (int i = 0; i < innerWeights.length; i++) {
            for (int j = 0; j < innerWeights[i].length; j++) {
                innerWeights[i][j] = Math.random();
            }
        }
    }

    public void train(int[][] trainingSetFeatures, int[] trainingSetLabels) {
        for (int j = 0; j < epochs; j++) { //iterations
            for (int i = 0; i < trainingSetLabels.length; i++) { //images
                forwardPropagation(trainingSetFeatures[i]);
                double mse = calcMeanSquareError(trainingSetLabels[i]);
                if (mse < 0.1) {
                    continue;
                }
                backwardPropagation(trainingSetFeatures[i], trainingSetLabels[i]);
            }
        }
    }

    private void forwardPropagation(int[] trainingSetFeature) {
        hiddenLayerValues = multiplyMatrices(weights, trainingSetFeature);
        performSigmoid(hiddenLayerValues);
        outputValues = multiplyMatrices(innerWeights, hiddenLayerValues);
        performSigmoid(outputValues);
    }

    private double[] multiplyMatrices(double[][] weights, int[] trainingSetFeature) {
        double[] result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int z = 0; z < trainingSetFeature.length; z++) {
                result[i] += weights[i][z] * trainingSetFeature[z];
            }
        }
        return result;
    }

    private double[] multiplyMatrices(double[][] weights, double[] hiddenLayerValues) {
        double[] result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int z = 0; z < hiddenLayerValues.length; z++) {
                result[i] += weights[i][z] * hiddenLayerValues[z];
            }
            result[i] %= 7;
        }
        return result;
    }

    private void performSigmoid(double[] hiddenLayerValues) {
        for (int i = 0; i < hiddenLayerValues.length; i++) {
            double exp = Math.exp(hiddenLayerValues[i] * -1.0);
            hiddenLayerValues[i] = 1.0 / (1.0 + exp);
        }
    }

    private double calcMeanSquareError(int trainingSetLabel) {
        double mse = Math.abs(trainingSetLabel - outputValues[0]);
        return mse;
    }

    private void backwardPropagation(int[] trainingSetFeatures, int trainingSetLabel) {
        double outputError = calcOutputError(trainingSetLabel);
        //a + (err*Wa1)
        this.newInnerWeights = new double[numOfOutputs][numOfNeurals];
        for (int i = 0; i < numOfNeurals; i++) {
            newInnerWeights[0][i] = innerWeights[0][i] + (outputError * hiddenLayerValues[i]);
        }
        double[] hiddenErrors = calcHiddenError(outputError);
        updateWeights(trainingSetFeatures, hiddenErrors);
    }

    private double calcOutputError(int trainingSetLabel) {
        //ErrorB = OutputB(1-OutputB)(TargetB – OutputB)
        double err = outputValues[0] * (1 - outputValues[0]) * (trainingSetLabel - outputValues[0]);
        return err;
    }

    private double[] calcHiddenError(double outputError) {
        //ErrorA = OutputA(1 - OutputA)(ErrorB WAB + Error c WAC)
        double[] errs = new double[numOfNeurals];
        for (int i = 0; i < numOfNeurals; i++) {
            errs[i] = hiddenLayerValues[i] * (1 - hiddenLayerValues[i]) * (outputError * innerWeights[0][i]);
        }
        return errs;
    }

    private void updateWeights(int[] trainingSetFeatures, double[] hiddenErrors) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = weights[i][j] + (n * hiddenErrors[i] * trainingSetFeatures[i]);
            }
        }
        for (int i = 0; i < innerWeights.length; i++) {
            System.arraycopy(newInnerWeights[i], 0, innerWeights[i], 0, innerWeights[i].length);
        }
    }

    public int[] predict(int[][] sampleImagesFeatures) {
        int[] resultLabels = new int[sampleImagesFeatures.length];
        for (int i = 0; i < sampleImagesFeatures.length; i++) {
            forwardPropagation(sampleImagesFeatures[i]);
            resultLabels[i] = (int) Math.round(outputValues[0]);
        }
        return resultLabels;
    }

    public double calculateAccuracy(int[] predictedLabels, int[] testingSetLabels) {
        int cnt = 0;
        for (int i = 0; i < testingSetLabels.length; i++) {
            if (predictedLabels[i] == testingSetLabels[i]) {
                cnt++;
            }
        }
        double res = ((double) cnt / testingSetLabels.length) * 100.0;
        return res;
    }

    public void save(String path) {
        JsonService jsonService = new JsonService();
        initNeuralModel();
        String jsonString = jsonService.toJson(neuralModel);
        try {
            FileWriter myWriter = new FileWriter(path);
            myWriter.write(jsonString);
            myWriter.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public NeuralNetwork load(String path) {
        JsonService jsonService = new JsonService();
        try {
            String jsonString = Files.readString(Paths.get(path));
            NeuralModel neuralModel = jsonService.fromJson(jsonString, NeuralModel.class);
            return initNeuralNetwork(neuralModel);
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return null;
        }

    }

    private NeuralNetwork initNeuralNetwork(NeuralModel neuralModel) {
        NeuralNetwork resultedNeuralNetwork = new NeuralNetwork();
        resultedNeuralNetwork.weights = new double[neuralModel.weights.length][neuralModel.weights[0].length];
        resultedNeuralNetwork.innerWeights = new double[neuralModel.innerWeights.length][neuralModel.innerWeights[0].length];
        for (int i = 0; i < neuralModel.weights.length; i++) {
            System.arraycopy(neuralModel.weights[i], 0, resultedNeuralNetwork.weights[i], 0, neuralModel.weights[i].length);
        }
        for (int i = 0; i < neuralModel.innerWeights.length; i++) {
            System.arraycopy(neuralModel.innerWeights[i], 0, resultedNeuralNetwork.innerWeights[i], 0, neuralModel.innerWeights[i].length);
        }
        return resultedNeuralNetwork;
    }

    public int predict(int[] sampleImgFeatures) {
        forwardPropagation(sampleImgFeatures);
        int result = (int) Math.round(outputValues[0]);
        return result;
    }
}