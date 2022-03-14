public class NeuralModel {

    public double[][] weights;
    public double[][] innerWeights;

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public void setInnerWeights(double[][] innerWeights) {
        this.innerWeights = innerWeights;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[][] getInnerWeights() {
        return innerWeights;
    }

    public NeuralModel()
    {

    }
}
