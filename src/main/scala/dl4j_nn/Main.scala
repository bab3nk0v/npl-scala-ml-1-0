package dl4j_nn

import java.io.File

import dl4j_helpers.PlotUtil
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.learning.config.{Nadam, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

object Main {

  val log: Logger = LoggerFactory.getLogger(Main.getClass)



  def main(args: Array[String]): Unit = {


    val filenameTrain = new ClassPathResource("/train.csv").getFile.getPath
    val filenameTest = new ClassPathResource("/test.csv").getFile.getPath


    val seed = 123
    val learningRate = 0.005
    val batchSize = 50
    val nEpochs = 100

    val numInputs = 2
    val numOutputs = 1
    val numHiddenNodes = 50



    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new File(filenameTrain)))
    var trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 1)


    val rrTest = new CSVRecordReader
    rrTest.initialize(new FileSplit(new File(filenameTest)))
    var testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 1)




    plotResults(filenameTrain, filenameTest, model, batchSize)

  }


  def plotResults(filenameTrain: String, filenameTest: String, model: MultiLayerNetwork, batchSize: Int): Unit = {
    val xMin = -1.5
    val xMax = 2.5
    val yMin = -1
    val yMax = 1.5

    //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
    val nPointsPerAxis = 100

    val evalPoints: Array[Array[Double]] = Array.ofDim[Double](nPointsPerAxis * nPointsPerAxis, 2)
    var count = 0

    for(i <- 1 to nPointsPerAxis) {
      for(j <- 1 to nPointsPerAxis) {
        val x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin
        val y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin
        evalPoints(count)(0) = x
        evalPoints(count)(1) = y
        count += 1
      }
    }

    val allXYPoints: INDArray = Nd4j.create(evalPoints)
    val predictionsAtXYPoints: INDArray = model.output(allXYPoints)


    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new File(filenameTrain)))
    var trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 1)

    //Load the test/evaluation data:
    val rrTest = new CSVRecordReader
    rrTest.initialize(new FileSplit(new File(filenameTest)))
    var testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 1)

    //Get all of the training data in a single array, and plot it:
    rr.initialize(new FileSplit(new ClassPathResource("/train.csv").getFile))
    rr.reset()
    val nTrainPoints = 2000
    trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 1)
    var ds = trainIter.next
    PlotUtil.plotTrainingData(ds.getFeatures, ds.getLabels, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

    //Get test data, run the test data through the network to generate predictions, and plot those predictions:
    rrTest.initialize(new FileSplit(new ClassPathResource("/test.csv").getFile))
    rrTest.reset()
    val nTestPoints = 1000
    testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 1)
    ds = testIter.next
    val testPredicted = model.output(ds.getFeatures)
    PlotUtil.plotTestData(ds.getFeatures, ds.getLabels, testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

  }

}