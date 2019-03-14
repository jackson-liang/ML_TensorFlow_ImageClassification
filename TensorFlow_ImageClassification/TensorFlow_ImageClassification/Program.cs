using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;

namespace TensorFlow_ImageClassification
{
    class Program
    {
        static readonly string AssetsFolder = @"D:\StepByStep\MyProjects\TestML\ML_TensorFlow_ImageClassification\Assets";
        static readonly string TrainDataFolder = Path.Combine(AssetsFolder,"train");
        static readonly string TrainTagsPath = Path.Combine(AssetsFolder, "train","tags.tsv");
        static readonly string TestDataFolder = Path.Combine(AssetsFolder, "test");
        static readonly string inceptionPb = Path.Combine(AssetsFolder, "inception", "tensorflow_inception_graph.pb");
        static readonly string imageClassifierZip = Path.Combine(Environment.CurrentDirectory,  "MLModel", "imageClassifier.zip");
               

        static void Main(string[] args)
        {
            //TrainAndSave();
            LoadAndPrediction();   

            Console.WriteLine("Press any to exit!");
            Console.ReadKey();
        }

        static void TrainAndSave()
        {
            MLContext mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.ReadFromTextFile<ImageNetData>(path: TrainTagsPath, hasHeader: false);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelTokey", inputColumnName:"Label")
                           .Append(mlContext.Transforms.LoadImages(TrainDataFolder, ("ImageReal","ImagePath")))
                           .Append(mlContext.Transforms.Resize(outputColumnName: "ImageReal", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "ImageReal"))
                           .Append(mlContext.Transforms.ExtractPixels(new ImagePixelExtractorTransformer.ColumnInfo(name: "input", inputColumnName: "ImageReal", interleave: ImageNetSettings.channelsLast, offset: ImageNetSettings.mean)))
                           .Append(mlContext.Transforms.ScoreTensorFlowModel(modelLocation: inceptionPb, outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }))
                           .Append(mlContext.MulticlassClassification.Trainers.LogisticRegression(labelColumn: "LabelTokey", featureColumn: "softmax2_pre_activation"))
                           .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabelValue", DefaultColumnNames.PredictedLabel)));

            // Train the model           
            ITransformer model = pipeline.Fit(data);

            using (var f = new FileStream(imageClassifierZip, FileMode.Create))
                mlContext.Model.Save(model, f);

            Console.WriteLine("Model saved!");
        }

        static void LoadAndPrediction()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // Load the model
            ITransformer loadedModel;
            using (var f = new FileStream(imageClassifierZip, FileMode.Open))
                loadedModel = mlContext.Model.Load(f);

            var predictor = loadedModel.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(mlContext);

            DirectoryInfo testdir = new DirectoryInfo(TestDataFolder);
            foreach(var jpgfile in testdir.GetFiles("*.jpg"))
            {  
                ImageNetData image = new ImageNetData();
                image.ImagePath = jpgfile.FullName;
                var pred = predictor.Predict(image);               

                Console.WriteLine($"Filename:{jpgfile.Name}:\tPredict Result:{pred.PredictedLabelValue}");
            }           
        }
    }
}
