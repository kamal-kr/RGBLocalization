using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Features2D;
using System.IO;
using System.Drawing;
using Emgu.CV.Structure;
using RGBLocalization.Utility;
using Emgu.CV.Util;
using Emgu.CV;
using MathNet.Numerics.LinearAlgebra.Double;

namespace RGBLocalization
{
    public class ImageFeatureExtraction
    {
        public class FeatureExtractionOptions
        {
            public FeatureExtractionOptions()
            {
                threshold = 10;
                numPoints = 100;
                featureExtractor = FastFeatureExtRaw;
            }

            public IEnumerable<MKeyPoint> DoExtract(Emgu.CV.Image<Gray, byte> image)
            {
                return featureExtractor(image, this);
            }

            public int threshold { get; set; }
            public int numPoints { get; set; }
            public FeatureExtractor featureExtractor { get; set; }
        }

        public delegate IEnumerable<MKeyPoint> FeatureExtractor(Emgu.CV.Image<Gray, byte> image, FeatureExtractionOptions options);

        public delegate Matrix<byte> FeatureDescriptionExtractor(Emgu.CV.Image<Gray, byte> im, MKeyPoint[] kp);
        
        public static IEnumerable<MKeyPoint> FastFeatureExt(Emgu.CV.Image<Gray, byte> image, FeatureExtractionOptions options)
        {
            return new FastDetector(options.threshold, true)
                                .DetectKeyPoints(image, (Emgu.CV.Image<Gray, byte>)null).ToArray()
                                .OrderByDescending(kp => kp.Response)
                                .Take(options.numPoints);
        }

        public static IEnumerable<MKeyPoint> FastFeatureExtRaw(Emgu.CV.Image<Gray, byte> image, FeatureExtractionOptions options)
        {
            return new FastDetector(options.threshold, true)
                                .DetectKeyPointsRaw(image, (Emgu.CV.Image<Gray, byte>) null).ToArray()
                                .OrderByDescending(kp => kp.Response)
                                .Take(options.numPoints);
        }

        public static Matrix<byte> ExtractBriefFeatureDescriptors(Emgu.CV.Image<Gray, byte> im, MKeyPoint[] kp)
        {
            var f = new VectorOfKeyPoint();
            f.Push(kp);
            return new BriefDescriptorExtractor().ComputeDescriptorsRaw(im, (Emgu.CV.Image<Gray, byte>)null, f);
        }

    }
}
