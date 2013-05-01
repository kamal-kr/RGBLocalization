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
    public class ImageFeatureMatching
    {
        public class FeatureMatchingOptions
        {
            public FeatureMatchingOptions()
            {
                featurePairDistanceThreshold = 32;
                featureDescriptionExtractor = ImageFeatureExtraction.ExtractBriefFeatureDescriptors;
            }
            public ImageFeatureExtraction.FeatureDescriptionExtractor featureDescriptionExtractor { get; set; }
            public MatrixDistance<int, byte> distanceFunction { get; set; }
            public int featurePairDistanceThreshold {get; set;}
        }
        
        public delegate DType MatrixDistance<DType, MType>(Emgu.CV.Matrix<MType> mat1, Emgu.CV.Matrix<MType> mat2)
            where MType : new()
            where DType : IComparable, new();

        public static int HammingDist(Emgu.CV.Matrix<byte> mat1, Emgu.CV.Matrix<byte> mat2)
        {
            return mat1.EnumerateRowwise().Zip(mat2.EnumerateRowwise(), (f, s) => f != s).Count(diff => diff);
        }

        public static IEnumerable<RType> NNMatchBruteForce<DType, MType, RType>(
        Emgu.CV.Matrix<MType> mat1,
        Emgu.CV.Matrix<MType> mat2,
        MatrixDistance<DType, MType> computeDistance,
        DType maxDistance,
        Func<int, int, DType, RType> mapToReturnType)
            where MType : new()
            where DType : IComparable, new()
        {
            return
                mat1.EnumerableRows()
                .Select((r, i) => mat2.EnumerableRows()
                                        .Select((rInner, iInner) => new { i, dist = computeDistance(r, rInner), iInner })
                                        .Aggregate(
                                                    new { i, dist = maxDistance, iInner = -1 },
                                                    (a, inner) => inner.dist.CompareTo(a.dist) < 0 ? inner : a)
                        )
                .Select(di => mapToReturnType(di.i, di.iInner, di.dist));
        }
    }
}
