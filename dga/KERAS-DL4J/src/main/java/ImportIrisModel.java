/**
 * Created by ylzhao on 2/8/17.
 */
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


public class ImportIrisModel {

    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model =
                org.deeplearning4j.nn.modelimport.keras.Model.importSequentialModel
                        ("/Users/ylzhao/Documents/dga/src/main/resources/iris_model_json",
                "/Users/ylzhao/Documents/dga/src/main/resources/iris_model");

        int numLinesToSkip = 0;
        String delimiter = ",";
        // Read the iris.txt file as a collection of records
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        // label index
        int labelIndex = 4;
        // num of classes
        int numClasses = 3;
        // batchsize all
        int batchSize = 150;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);

        DataSet allData = iterator.next();
        allData.shuffle();

        // Have our model
        //we have our Record Reader to read data
        // Evaluate the model

        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(allData.getFeatureMatrix());
        eval.eval(allData.getLabels(),output);
        System.out.println(eval.stats());
    }
}
