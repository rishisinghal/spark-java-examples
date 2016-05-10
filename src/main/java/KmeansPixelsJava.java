import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.catalyst.expressions.GenericRow;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import nodes.CachingNode;
import nodes.CifarParser;
import nodes.ClassLabelIndicatorsFromIntLabels;
import nodes.GrayScaler;
import nodes.ImageExtractor;
import nodes.LabelExtractor;
import nodes.LabeledImage;
import nodes.Vectorizer;
import scala.Function1;
import scala.Tuple2;
import scala.runtime.BoxedUnit;

/**
 * Trying K-Means on the CIFAR train dataset assuming it's not labelled.
 * We used the same basic feature extraction as we did on LinearPixelsJava
 * and see how much accuracy we get.
 * 
 */
public class KmeansPixelsJava {

	public static void main(String[] args) throws Exception {

		// Setup Spark context
		SparkConf conf = new SparkConf().setAppName("KmeansPixelsJava");
		conf.setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SparkContext sc = JavaSparkContext.toSparkContext(jsc);
		SQLContext sqlContext = new SQLContext(sc);

		// Read the cifar train data
		String trainDataFile = "C:\\rishi\\Big Data\\ampcamp\\ampcamp-pipelines\\data\\cifar_test.bin";

		// Read the data files
		RDD<LabeledImage> trainIimgRdd = new CifarParser().apply(new Tuple2<SparkContext, String>(sc, trainDataFile));
		RDD<LabeledImage> trainData = new CachingNode<LabeledImage>("TrainImageCache").apply(trainIimgRdd);
		RDD<float[]> trainFeatures = Vectorizer.apply(GrayScaler.apply(ImageExtractor.apply(trainData)));
	    RDD<float[]> trainLabels = new ClassLabelIndicatorsFromIntLabels(10).apply(LabelExtractor.apply(trainData));
	    
	    // Check the total number of images in the dataset
	    JavaRDD<LabeledImage> trainDataRdd = trainData.toJavaRDD();
	    System.out.println("Total Images:"+trainDataRdd.count());
	    
	    System.out.println("First image:"+trainDataRdd.first().toString());

	    trainLabels.toJavaRDD().foreach(f-> {String actualLabel="";int index=1; 
	    	for(float fl : f) 
	    	{
	    		if(fl==1.0)
	    		{
	    			actualLabel = actualLabel+" "+index;
	    		}
	    		index++;
	    	}
	    	System.out.println(actualLabel);
	    });
	    
		// Train the k-means model
		JavaRDD<float[]> jTrainFeatures = trainFeatures.toJavaRDD();
		JavaRDD<Row> points = jTrainFeatures.map(new ParsePoint());
		StructField[] fields = {new StructField("features", new VectorUDT(), false, Metadata.empty())};
		StructType schema = new StructType(fields);
		DataFrame dataset = sqlContext.createDataFrame(points, schema);

		KMeans kmeans = new KMeans().setK(10);
		KMeansModel model = kmeans.fit(dataset);

		// Shows the cluster centers
		Vector[] centers = model.clusterCenters();
		System.out.println("Cluster Centers: ");
		for (Vector center: centers) {
			System.out.println(center);
		}
		
		// How good did it perform?
		model.transform(dataset).write().format("json").save("cluster-out");
		//model.transform(dataset).select("prediction").show(10000); 
		//model.transform(dataset).select("prediction").write().format("json").save("cluster-out");
	}

	private static class ParsePoint implements Function<float[], Row> {
		@Override
		public Row call(float[] input) {
			double[] output = new double[input.length];
			for (int i = 0; i < input.length; i++)
			{
				output[i] = input[i];
			}
			Vector[] points = {Vectors.dense(output)};
			return new GenericRow(points);
		}
	}

}