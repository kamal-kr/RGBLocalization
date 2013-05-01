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
    public class ImageFeatures
    {
        public class FeatureExtractionOptions
        {
            public FeatureExtractionOptions()
            {
                threshold = 10;
                numPoints = 100;
            }
            public int threshold { get; set; }
            public int numPoints { get; set; }
        }

        public class FeatureMatchingOptions
        {
            public FeatureMatchingOptions()
            {
                featurePairDistanceThreshold = 32;
                featureDescriptionExtractor = ExtractBriefFeatureDescriptors;
            }
            public FeatureDescriptionExtractor featureDescriptionExtractor { get; set; }
            public MatrixDistance<int, byte> distanceFunction { get; set; }
            public int featurePairDistanceThreshold {get; set;}
        }

        public class RGBMatchOptions<MType> 
            where MType : new()
        {
            public RGBMatchOptions()
            {
                allFeaturePairsAction = l => { };
                optimalFeaturePairsAction = l => { };
                matchingOptions = new FeatureMatchingOptions();
                ransacOptions = new SimpleRansac.RansacOptions();
            }

            public Func<Emgu.CV.Image<Gray, MType>, IEnumerable<MKeyPoint>> featureExtractor { get; set; }
            public FeatureMatchingOptions matchingOptions { get; set; }
            public SimpleRansac.RansacOptions ransacOptions { get; set; }
            public Action<IEnumerable<Tuple<PointF, PointF>>> allFeaturePairsAction { get; set; }
            public Action<IEnumerable<Tuple<PointF, PointF>>> optimalFeaturePairsAction { get; set; }
        }
        

        public delegate IEnumerable<MKeyPoint> FeatureExtractor(Emgu.CV.Image<Gray, byte> image, FeatureExtractionOptions options);

        public delegate Matrix<byte> FeatureDescriptionExtractor(Emgu.CV.Image<Gray, byte> im, MKeyPoint[] kp);
        
        public delegate DType MatrixDistance<DType, MType>(Emgu.CV.Matrix<MType> mat1, Emgu.CV.Matrix<MType> mat2)
            where MType : new()
            where DType : IComparable, new();

        public static int HammingDist(Emgu.CV.Matrix<byte> mat1, Emgu.CV.Matrix<byte> mat2)
        {
            return mat1.EnumerateRowwise().Zip(mat2.EnumerateRowwise(), (f, s) => f != s).Count(diff => diff);
        }
        
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
