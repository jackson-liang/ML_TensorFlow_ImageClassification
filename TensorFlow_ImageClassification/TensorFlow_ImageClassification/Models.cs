using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TensorFlow_ImageClassification
{
    public struct ImageNetSettings
    {
        public const int imageHeight = 224;
        public const int imageWidth = 224;
        public const float mean = 117;
        public const float scale = 1;
        public const bool channelsLast = true;
    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImageNetPrediction
    {
        public float[] Score;
        public string PredictedLabelValue;
    }
}
