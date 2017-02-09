import java.io.File;
import java.util.Scanner;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ylzhao on 2/1/17.
 */

public class ImportKerasModel{

    public static void main(String[] args) throws Exception{
        String inputfile = args[0];
        // Load Model
        ImportKerasModel obj = new ImportKerasModel();
        // Test input string
        obj.test(inputfile);
    }

    private String[] test(String inputfile) throws Exception{
        // Get prediction result
        String[] binary_labels = {"Benign", "Malicious"};
        String[] multi_labels = {"zeus", "corebot",
                "pushdo", "ramnit", "matsnu", "banjori", "tinba",
                "rovnix", "conficker", "locky", "cryptolocker"};

        String binarycols = "binarycols.txt";
        INDArray binarytest = getFeature(inputfile, binarycols);
        MultiLayerNetwork binary_model = binary();
        int[] binary_pred = binary_model.predict(binarytest);
        String binary_res = binary_labels[binary_pred[0]];

        String multicols = "multicols.txt";
        INDArray multitest = getFeature(inputfile, multicols);
        MultiLayerNetwork multi_model = multi();
        int[] multi_pred = multi_model.predict(multitest);
        String multi_res = multi_labels[multi_pred[0]];

        if (binary_pred[0] == 0){
            System.out.println("Safe domain address: " + inputfile + ";");
        } else{
            System.out.println("Malicious domain address: " + inputfile + ";");
            System.out.println("Suspicious Malware Type: " + multi_res + ";");
        }

        return new String[]{binary_res, multi_res};
    }

    private MultiLayerNetwork binary() throws Exception{
        // Import Keras Binary Model
        ClassLoader classLoader = getClass().getClassLoader();
        String modelJsonFilename = classLoader.getResource("binary_model_json").getFile();
        String weightsHdf5Filename = classLoader.getResource("binary_model").getFile();
        return Model.importSequentialModel(modelJsonFilename,weightsHdf5Filename);
    }

    private MultiLayerNetwork multi() throws Exception{
        // Import Keras Multi Model
        ClassLoader classLoader = getClass().getClassLoader();
        String modelJsonFilename = classLoader.getResource("multi_model_json").getFile();
        String weightsHdf5Filename = classLoader.getResource("multi_model").getFile();
        return Model.importSequentialModel(modelJsonFilename, weightsHdf5Filename);
    }

    private String getCols(String fileName)throws Exception{
        // Get cols from column files
        StringBuilder result = new StringBuilder("");
        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());
        Scanner scanner = new Scanner(file);
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            result.append(line).append("\n");
        }
        scanner.close();
        return result.toString();
    }

    private INDArray getFeature(String inputfile, String filename)throws Exception{
        // Get feature based on column files
        String result = getCols(filename);
        String[] inputlist = result.split("\n");
        float[] feature = new float[inputlist.length+1];
        for(int i=1; i<inputlist.length; i++){
            if(inputfile.contains(inputlist[i])){
                feature[i] = 1;
            }else{
                feature[i] = 0;
            }
        }
        if (inputfile.length() > 20) {
            feature[inputlist.length] = 1;
        } else {
            feature[inputlist.length] = 0;
        }
        return Nd4j.create(feature, new int[]{1,feature.length});
    }
}
