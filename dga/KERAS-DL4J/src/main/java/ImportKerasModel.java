import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;
/**
 * Created by ylzhao on 2/1/17.
 */

public class ImportKerasModel{

    public static void main(String[] args) throws Exception{
        // Load Model
        ImportKerasModel obj = new ImportKerasModel();
        String[] inputfile = args;
        obj.test(inputfile);
    }

    private ArrayList test(String[] inputfile) throws Exception{
        // Get prediction result

        String[] binary_labels = {"Benign", "Malicious"};
        String[] multi_labels = {"zeus", "corebot", "goz",
                "pushdo", "ramnit", "matsnu", "banjori", "tinba",
                "rovnix", "conficker", "locky", "cryptolocker"};

        ArrayList output = new ArrayList<Tuple2>();

        String binarycols = "binarycols.txt";
        INDArray binarytest = getFeature(inputfile, binarycols);
        MultiLayerNetwork binary_model = binary();
        int imax1 = Nd4j.getBlasWrapper().iamax(binary_model.output(binarytest).mean(0));
        INDArray binary_mean = binary_model.output(binarytest).mean(0);

        String multicols = "multicols.txt";
        INDArray multitest = getFeature(inputfile, multicols);
        MultiLayerNetwork multi_model = multi();
        INDArray multi_mean = multi_model.output(multitest).mean(0);

        output.add(new Tuple2<String, Float>(binary_labels[imax1],binary_mean.getFloat(imax1)));
        for(int i=0; i < multi_labels.length; i++){
            output.add(new Tuple2<String, Float>(multi_labels[i],multi_mean.getFloat(i)));
        }
        /*
        for(int i = 0; i < inputfile.length; i++){
            String binary_res = binary_labels[binary_pred[i]];
            String multi_res = multi_labels[multi_pred[i]];
            if (binary_pred[i] == 0){
                System.out.println("Safe domain address: " + inputfile[i] + ";");
            } else{
                System.out.println("Malicious domain address: " + inputfile[i] + ";");
                System.out.println("Suspicious Malware Type: " + multi_res + ";");
            }
        }
        */
        System.out.println(output);
        return output;
    }

    private int[] count(int[] pred_list){
        int[] counter = new int[pred_list.length];
        for(int i=0; i<pred_list.length; i++){
            counter[pred_list[i]]++;
        }
        return counter;
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
            if (inputstring.length() > 15) {
                featureArray[++nextIndex *(y+1)-1] = 1;
            } else {
                featureArray[++nextIndex *(y+1)-1] = 0;
            }
        }

        return Nd4j.create(featureArray, new int[]{x, y+1});
    }
}
