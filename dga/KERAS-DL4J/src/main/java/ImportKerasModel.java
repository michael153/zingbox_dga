import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ylzhao on 2/1/17.
 */

public class ImportKerasModel{

    public static void main(String[] args) throws Exception{
        // Load Model
        ImportKerasModel obj = new ImportKerasModel();
        String[] inputfile = args;
        obj.evaluate(inputfile);
    }

    private String[] evaluate(String[] inputfile) throws Exception{
        // Get prediction result

        String[] binary_labels = {"Benign", "Malicious"};
        String[] multi_labels = {"zeus", "corebot", "goz",
                "pushdo", "ramnit", "matsnu", "banjori", "tinba",
                "rovnix", "conficker", "locky", "cryptolocker"};

        String binarycols = "binarycols.txt";
        INDArray binarytest = getFeature(inputfile, binarycols);
        MultiLayerNetwork binary_model = binary();
        int[] binary_pred = binary_model.predict(binarytest);

        String multicols = "multicols.txt";
        INDArray multitest = getFeature(inputfile, multicols);
        MultiLayerNetwork multi_model = multi();
        int[] multi_pred = multi_model.predict(multitest);

        String[] output = new String[inputfile.length];

        for(int i=0; i < inputfile.length; i++){
            String binary_res = binary_labels[binary_pred[i]];
            String multi_res = multi_labels[multi_pred[i]];
            if (binary_pred[i] == 0){
                System.out.println("Safe domain address: " + inputfile[i] + ";");
                output[i] = binary_res;
            } else{
                System.out.println("Malicious domain address: " + inputfile[i] + ";");
                System.out.println("Suspicious Malware Type: " + multi_res + ";");
                output[i] = multi_res;
            }
        }
        return output;
    }

    public MultiLayerNetwork binary() throws Exception{
        // Import Keras Binary Model
        ClassLoader classLoader = getClass().getClassLoader();
        String modelJsonFilename = classLoader.getResource("binary_model_json").getFile();
        String weightsHdf5Filename = classLoader.getResource("binary_model").getFile();
        return Model.importSequentialModel(modelJsonFilename,weightsHdf5Filename);
    }

    public MultiLayerNetwork multi() throws Exception{
        // Import Keras Multi Model
        ClassLoader classLoader = getClass().getClassLoader();
        String modelJsonFilename = classLoader.getResource("multi_model_json").getFile();
        String weightsHdf5Filename = classLoader.getResource("multi_model").getFile();
        return Model.importSequentialModel(modelJsonFilename, weightsHdf5Filename);
    }

    private String getCols(String fileName)throws Exception{
        // Get cols from column files
        StringBuilder result = new StringBuilder("");
        String line = null;
        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        BufferedReader bufferedReader = new BufferedReader(new FileReader(classLoader.getResource(fileName).getFile()));
        while ((line = bufferedReader.readLine()) != null)
        {
            result.append(line).append("\n");
        }
        //scanner.close();
        bufferedReader.close();
        return result.toString();
    }


    private INDArray getFeature(String[] inputfile, String filename)throws Exception{

        // Get feature based on column files

        String result = getCols(filename);
        String[] cols = result.split("\n");
        int x = inputfile.length;
        int y = cols.length;
        float[] featureArray = new float[(y+1)*x];
        int nextIndex = 0;

        for(String inputstring : inputfile){
            for(int i=0; i<y; i++){
                if(inputstring.contains(cols[i])){
                    featureArray[(nextIndex)*(y+1)+i] = 1;
                }else{
                    featureArray[(nextIndex)*(y+1)+i] = 0;
                }
            }
            if (inputstring.length() > 20) {
                featureArray[++nextIndex *(y+1)-1] = 1;
            } else {
                featureArray[++nextIndex *(y+1)-1] = 0;
            }
        }

        return Nd4j.create(featureArray, new int[]{x, y+1});
    }
}
