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

namespace RGBLocalization
{
    class RGBMatch
    {
        public static string imageMapDir = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\";
        public static string workingDir = @"C:\Kamal\RSE\WorkingDirs\OpenCVTrial";

        static void Main(string[] args)
        {
           //RGBMatchTests.CmdLineImageSearch(Directory.GetFiles(imageMapDir, "*image.png"));
            //ExtractFastFeatures(@"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389268-657175217-image.png");
            //RGBMatchTests.DebugRGBMatch();
            Pose3DTests.DoTest();


           Console.WriteLine("Done! Hit enter!");
           Console.ReadLine();
        }

        public delegate DType MatrixDistance<DType, MType>(Emgu.CV.Matrix<MType> mat1, Emgu.CV.Matrix<MType> mat2)
            where MType : new()
            where DType : IComparable, new();

        public static MatrixDistance<int, byte> HammingDist =
                (mat1, mat2) => mat1.EnumerateRowwise().Zip(mat2.EnumerateRowwise(), (f, s) => f != s).Count(diff => diff);

        public delegate IEnumerable<MKeyPoint> FeatureExtractor<MType, MImType>(Emgu.CV.Image<Gray, MImType> image, int threshold, int numPoints)
            where MImType : new();

        public static FeatureExtractor<byte, byte> FastFeatureExt =
            (im, threshold, numPoints) => new FastDetector(threshold, true).DetectKeyPoints(im, null).OrderByDescending(kp => kp.Response).Take(numPoints);

        public static FeatureExtractor<byte, byte> FastFeatureExtRaw =
            (im, threshold, numPoints) => new FastDetector(threshold, true).DetectKeyPointsRaw(im, null).ToArray().OrderByDescending(kp => kp.Response).Take(numPoints);

        public delegate Matrix<MType> FeatureDescriptor<MType>(Emgu.CV.Image<Gray, MType> im, MKeyPoint[] kp)
            where MType : new();

        public static FeatureDescriptor<byte> BriefFeatureDescriptorFunc =
            (im, kp) =>
            {
                var f = new VectorOfKeyPoint();
                f.Push(kp);
                return new BriefDescriptorExtractor().ComputeDescriptorsRaw(im, (Emgu.CV.Image<Gray, byte>)null, f);
            };


        public static IEnumerable<RType> NNMatchBruteForce<DType, MType, RType>(
                Emgu.CV.Matrix<MType> mat1, 
                Emgu.CV.Matrix<MType> mat2, 
                MatrixDistance<DType, MType> computeDistance,
                DType maxDistance,
                Func<int, int, DType, RType> mapToReturnType)
            where MType: new()
            where DType: IComparable, new()
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
                /*
                mat1.EnumerableRows()
                .SelectMany((r, i) => mat2.EnumerableRows()
                                        .Select((rInner, iInner) => new { i, dist = computeDistance(r, rInner), iInner })
                                        .Where(p => p.dist.CompareTo(maxDistance) < 0)
                                        .Select(p => new { p.i, p.iInner, p.dist}))
                .Select(di => mapToReturnType(di.i, di.iInner, di.dist));*/
        }
        
        
        public static double DoRGBMatch(
                        Emgu.CV.Image<Gray, byte> image1, 
                        Emgu.CV.Image<Gray, byte> image2, 
                        FeatureExtractor<byte, byte> featExt,
                        int featExtThreshold,
                        int numFeaturePoints,
                        FeatureDescriptor<byte> featDesc,
                        MatrixDistance<int, byte> distanceFunction,
                        int featurePairDistanceThreshold,
                        double ransacSqErrThresh,
                        int ransacNumTrials,
                        Random rand,
                        Func<IEnumerable<Tuple<PointF, PointF>>, int> featurePairCandidatesInspector,
                        Func<IEnumerable<Tuple<PointF, PointF>>, int> optimalFeaturePairsInspector
                        )
        {
            
            //WARNING!!
            Console.WriteLine("extracting only 2 point from the query image!");
            var features1 = featExt(image1, featExtThreshold, 20).ToArray();
            
            
            
            var features2 = featExt(image2, featExtThreshold, numFeaturePoints).ToArray();

            var featurePairs =
                NNMatchBruteForce(
                    featDesc(image1, features1),
                    featDesc(image2, features2),
                    distanceFunction,
                    int.MaxValue,
                    (iFrom, iTo, dist) => new { dist, pair = new Tuple<PointF, PointF>(features1[iFrom].Point, features2[iTo].Point) })
                .Where(p => p.dist < featurePairDistanceThreshold)
                .Select(p => p.pair)
                .ToList();

            featurePairCandidatesInspector(featurePairs);

            var optimalPairs = SimpleRansac.RansacMatch(featurePairs, ransacNumTrials, 3, 6, ransacSqErrThresh, rand);

            if (optimalPairs.Item2 != null)
            {
                optimalFeaturePairsInspector(optimalPairs.Item2);
            }

            return optimalPairs.Item1;
        }

        public static int VisualizeFeaturePairs(IEnumerable<Tuple<PointF, PointF>> featurePairs, string image1, string image2, string fileName)
        {
            var im1 = new Emgu.CV.Image<Rgb, byte>(image1);
            var im2 = new Emgu.CV.Image<Rgb, byte>(image2);
            
          
            
            featurePairs
            .Aggregate(
                    im1.ConcateHorizontal(im2),
                    (a, i) =>
                    {
                        var traslatedPoint2 = new PointF((i.Item2.X + im1.Width), i.Item2.Y);
                        a.Draw(new CircleF(i.Item1, 5), new Rgb(10, 10, 10), 1);
                        a.Draw(new CircleF(traslatedPoint2, 5), new Rgb(10, 10, 10), 1);
                        a.Draw(new LineSegment2DF(i.Item1, traslatedPoint2), new Rgb(20, 20, 20), 2);
                        a.Save(fileName);
                        return a;
                    });


            return 0;
        }

        static void MatchFastFeatures()
        {
            var fastFeatures =
                Directory.GetFiles(imageMapDir, "*image.png")
                //.Skip(10)
                .Take(2)
                .Select(img => new { img, grayImage = new Emgu.CV.Image<Gray, byte>(img) })
                .Select(img => new { img, features = new FastDetector(100, true).DetectKeyPointsRaw(img.grayImage, null) })
                .Select(feat => new
                {
                    feat.img,
                    feat.features,
                    briefDesc = new BriefDescriptorExtractor().ComputeDescriptorsRaw(new Emgu.CV.Image<Gray, byte>(feat.img.img), (Emgu.CV.Image<Gray, byte>)null, feat.features)
                })
                .ToList();
            
            Console.WriteLine(fastFeatures[0].briefDesc.Height);

            
            NNMatchBruteForce(
                fastFeatures[0].briefDesc,
                fastFeatures[1].briefDesc,
                HammingDist,
                int.MaxValue,
                (iFrom, iTo, dist) => new { point1 = fastFeatures[0].features[iFrom].Point, point2 = fastFeatures[1].features[iTo].Point, dist })
            .OrderBy(p => p.dist)
            .Select((p, i) => new { p, i})
            .Aggregate(
            new Emgu.CV.Image<Rgb, byte>(fastFeatures[0].img.img)
                    .ConcateHorizontal(new Emgu.CV.Image<Rgb, byte>(fastFeatures[1].img.img)),
                    (a, i) =>
                    {
                        var traslatedPoint2 = new PointF((i.p.point2.X + fastFeatures[0].img.grayImage.Width), i.p.point2.Y);
                        a.Draw(new CircleF(i.p.point1, 5), new Rgb(10, 10, 10), 1);
                        a.Draw(new CircleF(traslatedPoint2, 5), new Rgb(10, 10, 10), 1);
                        a.Draw(new LineSegment2DF(i.p.point1, traslatedPoint2), new Rgb(20, 20, 20), 2);
                        a.Save(Path.Combine(workingDir, "sbs_" + i.i.ToString() + ".png"));
                        return a;
                    });
        }


        static void ExtractFastFeatures(string sourceImage)
        {
            Emgu.CV.Image<Emgu.CV.Structure.Gray, byte> i = new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(sourceImage);

                new FastDetector(30, true).DetectKeyPointsRaw(i, i).ToArray()
                .OrderByDescending(kp => kp.Response)
                //.Select(kk => new { kk.Point, batchID = kk. }))
                .Select((kp, index) => new { kp.Point, batchID = index / 10 })
                .GroupBy(kp => kp.batchID)
                .Select(batch =>
                {
                    batch
                   .Aggregate(
                            new Emgu.CV.Image<Emgu.CV.Structure.Rgb, byte>(sourceImage),
                            (a, p) =>
                            {
                                a.Draw(new CircleF(p.Point, 5), new Rgb(10, 10, 10), 1);
                                return a;
                            })
                    .Save(Path.Combine(workingDir, "DetectKeyPointsRaw_" + batch.Key.ToString() + "_" + Path.GetFileName(sourceImage)));
                    return batch;
                })
                .Count();
        }

    }
}
