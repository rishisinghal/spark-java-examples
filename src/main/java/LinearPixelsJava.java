import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;

import nodes.CachingNode;
import nodes.CifarParser;
import nodes.ClassLabelIndicatorsFromIntLabels;
import nodes.GrayScaler;
import nodes.ImageExtractor;
import nodes.LabelExtractor;
import nodes.LabeledImage;
import nodes.LinearMapper;
import nodes.Vectorizer;
import scala.Tuple2;
import utils.Stats;

/**
 * Takes the LinearPixels Scala example present 
 * at http://ampcamp.berkeley.edu/5/exercises/image-classification-with-pipelines.html#setup
 * and does the same without pipelines in Java
 */
public class LinearPixelsJava {

  public static void main(String[] args) throws Exception {

	// Setup Spark context to run it locally  
    SparkConf conf = new SparkConf().setAppName("LinearPixelsJava");
    conf.setMaster("local");
    JavaSparkContext jsc = new JavaSparkContext(conf);
    SparkContext sc = JavaSparkContext.toSparkContext(jsc);
    
    // Read the cifar train & test data
    String trainDataFile = null;
    String testDataFile = null;
    
    if(args.length<2)
    {
    	trainDataFile = "C:\\rishi\\Big Data\\ampcamp\\ampcamp-pipelines\\data\\cifar_train.bin";
    	testDataFile = "C:\\rishi\\Big Data\\ampcamp\\ampcamp-pipelines\\data\\cifar_test.bin";
    } else {
    	trainDataFile = args[0]; 
    	testDataFile = args[1];
    }
    
    // Read the data files
    RDD<LabeledImage> trainIimgRdd = new CifarParser().apply(new Tuple2<SparkContext, String>(sc, trainDataFile));
    RDD<LabeledImage> trainData = new CachingNode<LabeledImage>("TrainImageCache").apply(trainIimgRdd);

    RDD<LabeledImage> testIimgRdd = new CifarParser().apply(new Tuple2<SparkContext, String>(sc, testDataFile));
    RDD<LabeledImage> testData = new CachingNode<LabeledImage>("TestImageCache").apply(testIimgRdd);

    // Check the total number of images in the dataset
    JavaRDD<LabeledImage> trainDataRdd = trainData.toJavaRDD();
    System.out.println("Total Images:"+trainDataRdd.count());
    
    // Create the train featurizer
    RDD<float[]> trainFeatures = Vectorizer.apply(GrayScaler.apply(ImageExtractor.apply(trainData)));
    RDD<float[]> trainLabels = new ClassLabelIndicatorsFromIntLabels(10).apply(LabelExtractor.apply(trainData));
    
    // Train the model & get the error
    LinearMapper model = LinearMapper.train(trainFeatures, trainLabels);    
    float trainError = Stats.classificationError(model.apply(trainFeatures), trainLabels, 1);
    System.out.println("Error on training data is:"+trainError); 
    
    // Test the model created
    // Create the test featurizer & test labels
    RDD<float[]> testFeatures = Vectorizer.apply(GrayScaler.apply(ImageExtractor.apply(testData)));
    RDD<float[]> testLabels = new ClassLabelIndicatorsFromIntLabels(10).apply(LabelExtractor.apply(testData));

    float testError = Stats.classificationError(model.apply(testFeatures), testLabels, 1);
    System.out.println("Error on test data is:"+testError);    
  }
}