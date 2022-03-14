public class ImageData {
    int[] pixels;
    int label;

    public ImageData(){}
    ImageData(int imgHeight, int imgWidth) { pixels = new int[imgHeight*imgWidth]; }
    public void setPixels(int[] pixels) { this.pixels = pixels; }
    public void setLabel(int lbl) { label = lbl; }
}