import java.io.Serializable;
import java.util.ArrayList;
import java.util.function.Function;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.jblas.FloatMatrix;

import nodes.CachingNode;
import nodes.CifarParser;
import nodes.ClassLabelIndicatorsFromIntLabels;
import nodes.Convolver;
import nodes.DownSampler;
import nodes.FeatureNormalize;
import nodes.GrayScaler;
import nodes.ImageExtractor;
import nodes.ImageVectorizer;
import nodes.InterceptAdder;
import nodes.LabelExtractor;
import nodes.LabeledImage;
import nodes.LinearMapper;
import nodes.Pooler;
import nodes.SymmetricRectifier;
import nodes.Vectorizer;
import nodes.ZCAWhitener;
import scala.Function1;
import scala.Option;
import scala.Tuple2;
import utils.Image;
import utils.Stats;
import utils.ZCAWhitenerExtended;

/**
 * Takes the RandomVocab Scala example present 
 * at http://ampcamp.berkeley.edu/5/exercises/image-classification-with-pipelines.html#setup
 * and does the same without pipelines in Java
 */
public class RandomVocabJava {

	public static void main(String[] args) throws Exception {

		// Setup Spark context to run it locally  
		SparkConf conf = new SparkConf().setAppName("RandomVocabJava");
		conf.setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SparkContext sc = JavaSparkContext.toSparkContext(jsc);

		// Setup the constants. Some can be overridden by arguments being passed
		String trainDataFile = null;
		String testDataFile = null;

		int numClasses = 10;
		int imageSize = 32;
		int numChannels = 3;

		int numFilters=200;
		int patchSize = 6;

		float alpha = 0.25f; 

		int poolSize = 14;
		int poolStride = 13;

		float lambda = 10; 
		double sample = 0.2;    
		boolean isSample = true;

		if(args.length<4)
		{
			System.out.println("Using default parameter values");
			trainDataFile = "C:\\rishi\\Big Data\\ampcamp\\ampcamp-pipelines\\data\\cifar_train.bin";
			testDataFile = "C:\\rishi\\Big Data\\ampcamp\\ampcamp-pipelines\\data\\cifar_test.bin";
		} else {
			trainDataFile = args[0]; 
			testDataFile = args[1];
			numFilters = Integer.parseInt(args[2]);
			lambda = Integer.parseInt(args[3]);
		}

		if(args.length > 4) {
			sample = Double.parseDouble(args[4]);
			isSample = true;
		}

		// Read the train data files
		RDD<LabeledImage> trainData = null;

		if(isSample)
			trainData = new CachingNode<LabeledImage>("TrainImageCache")
			.apply(new DownSampler<LabeledImage>(sample)
					.apply(new CifarParser()
							.apply(new Tuple2<SparkContext, String>(sc, trainDataFile))));
		else
			trainData = new CachingNode<LabeledImage>("TrainImageCache")
			.apply(new CifarParser()
					.apply(new Tuple2<SparkContext, String>(sc, trainDataFile)));

		// Check the total number of images in the dataset
		JavaRDD<LabeledImage> trainDataRdd = trainData.toJavaRDD();
		System.out.println("Total Images:"+trainDataRdd.count());

		// Create the train featurizer
		FloatMatrix filters = FloatMatrix.randn(numFilters, patchSize*patchSize*numChannels);
		float[][] filterArray = filters.toArray2();

		/*
    val featurizer =
    	      ImageExtractor
    	        .andThen(new Convolver(sc, filterArray, imageSize, imageSize, numChannels, None, true))
    	        .andThen(SymmetricRectifier(alpha=alpha))
    	        .andThen(new Pooler(poolStride, poolSize, identity, _.sum))
    	        .andThen(new ImageVectorizer)
    	        .andThen(new CachingNode)
    	        .andThen(new FeatureNormalize)
    	        .andThen(new InterceptAdder)
    	        .andThen(new CachingNode)
		 */

		RDD<Image> t = new SymmetricRectifier(0, alpha)
				.apply(new Convolver(sc, filterArray, imageSize, imageSize, numChannels, Option.apply(null), true, 10.0)
						.apply(ImageExtractor.apply(trainData)));

		RDD<float[]> trainFea = new CachingNode<float[]>("featureCache")
				.apply(new ImageVectorizer()
						.apply(new MyPooler(poolStride, poolSize)
								.apply(new SymmetricRectifier(0, alpha)
										.apply(new Convolver(sc, filterArray, imageSize, imageSize, numChannels, Option.apply(null), true, 10.0)
												.apply(ImageExtractor.apply(trainData))))));


		RDD<float[]> trainFeatures = new CachingNode<float[]>("featureCache1")
										.apply(new InterceptAdder()
												.apply(new FeatureNormalize(1e-12f, -1)
														.apply(trainFea)));

		RDD<float[]> trainLabels = new CachingNode<float[]>("labelCache")
				.apply(new ClassLabelIndicatorsFromIntLabels(numClasses)
						.apply(LabelExtractor.apply(trainData)));

		// Train the model & get the error
		LinearMapper model = LinearMapper.trainWithL2(trainFeatures, trainLabels, lambda);    
		float trainError = Stats.classificationError(model.apply(trainFeatures), trainLabels, 1);
		System.out.println("Error on training data is:"+trainError); 

		// Test the model created
		/*
    RDD<LabeledImage> testIimgRdd = new CifarParser().apply(new Tuple2<SparkContext, String>(sc, testDataFile));
    RDD<LabeledImage> testData = new CachingNode<LabeledImage>("TestImageCache").apply(testIimgRdd);

    // Create the test featurizer & test labels
    RDD<float[]> testFeatures = Vectorizer.apply(GrayScaler.apply(ImageExtractor.apply(testData)));
    RDD<float[]> testLabels = new ClassLabelIndicatorsFromIntLabels(numClasses).apply(LabelExtractor.apply(testData));

    float testError = Stats.classificationError(model.apply(testFeatures), testLabels, 1);
    System.out.println("Error on test data is:"+testError);
		 */    
	}
}


class pixelFunction implements Function1,Serializable{

	@Override
	public Function1 andThen(Function1 function1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object apply(Object arg0) {
		return arg0;
	}

	@Override
	public Function1 compose(Function1 function1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean apply$mcZD$sp(double d) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcID$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVD$sp(double d) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZF$sp(float f) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcIF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVF$sp(float f) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZI$sp(int i) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcII$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVI$sp(int i) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZJ$sp(long l) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcIJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVJ$sp(long l) {
		// TODO Auto-generated method stub

	}

}

class poolFunction implements Function1<float[], Object>,Serializable{

	@Override
	public <A> Function1<float[], A> andThen(Function1<Object, A> function1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object apply(float[] arg0) {
		float sum = 0;

		for(int i =0; i<arg0.length; i++)
			sum=sum+arg0[i];

		return sum;
	}

	@Override
	public <A> Function1<A, Object> compose(Function1<A, float[]> function1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean apply$mcZD$sp(double d) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcID$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJD$sp(double d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVD$sp(double d) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZF$sp(float f) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcIF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJF$sp(float f) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVF$sp(float f) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZI$sp(int i) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcII$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJI$sp(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVI$sp(int i) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean apply$mcZJ$sp(long l) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double apply$mcDJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float apply$mcFJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int apply$mcIJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long apply$mcJJ$sp(long l) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void apply$mcVJ$sp(long l) {
		// TODO Auto-generated method stub

	}

}