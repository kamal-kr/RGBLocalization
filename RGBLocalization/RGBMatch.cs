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
    public class RGBMatch
    {
        public class RGBMatchOptions<MType> 
            where MType : new()
        {
            public RGBMatchOptions()
            {
                allFeaturePairsAction = l => { };
                optimalFeaturePairsAction = l => { };
                matchingOptions = new ImageFeatureMatching.FeatureMatchingOptions();
                ransacOptions = new SimpleRansac.RansacOptions();
            }

            public Func<Emgu.CV.Image<Gray, MType>, IEnumerable<MKeyPoint>> featureExtractor { get; set; }
            public ImageFeatureMatching.FeatureMatchingOptions matchingOptions { get; set; }
            public SimpleRansac.RansacOptions ransacOptions { get; set; }
            public Action<IEnumerable<Tuple<PointF, PointF>>> allFeaturePairsAction { get; set; }
            public Action<IEnumerable<Tuple<PointF, PointF>>> optimalFeaturePairsAction { get; set; }
        }
        
        public static string imageMapDir = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\";
        public static string workingDir = @"C:\Kamal\RSE\WorkingDirs\OpenCVTrial";

        static void Main(string[] args)
        {
           //RGBMatchTests.CmdLineImageSearch(Directory.GetFiles(imageMapDir, "*image.png"));
            //ExtractFastFeatures(@"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\1302389115-238787139-image.png");
            //RGBMatchTests.DebugRGBMatch();
            //Pose3DTests.TestPosePrimitives();
            //ImageMapTests.TestImageMap();
            //ImageMapTests.TestImageMapConsistency(
            //                @"C:\Kamal\RSE\WorkingDirs\Visualizaton\espresso-1.bag.dump.map",
            //                @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt");
            //Pose3DTests.TestPoseEstimation();
            //Pose3DTests.TestPoseEstimationOpenCV();
            //Pose3DTests.TestPoseEstOpenCVRealData();
            CamLocalization.MatchRGBImage(@"C:\Kamal\RSE\WorkingDirs\Visualizaton\espresso-1.bag.dump.map",
                                           @"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389330-098953546-image.png",
                                           new ImageFeatureExtraction.FeatureExtractionOptions(),
                                           new Random(0));
            //Pose3DTests.TestOpenCVProjection();

           Console.WriteLine("Done! Hit enter!");
           Console.ReadLine();
        }

        
        
        /*public static double DoRGBMatch(
                        Emgu.CV.Image<Gray, byte> image1,
                        Emgu.CV.Image<Gray, byte> image2, 
                        RGBMatchOptions<byte> options,
                        Random rand)
        {
            var features1 = options.featureExtractor(image1).ToArray();
            var features2 = options.featureExtractor(image2).ToArray();

            var featurePairs =
                ImageFeatureMatching.NNMatchBruteForce(
                    options.matchingOptions.featureDescriptionExtractor(image1, features1),
                    options.matchingOptions.featureDescriptionExtractor(image2, features2),
                    options.matchingOptions.distanceFunction,
                    options.matchingOptions.featurePairDistanceThreshold,
                    (iFrom, iTo, dist) => new { dist, pair = new Tuple<PointF, PointF>(features1[iFrom].Point, features2[iTo].Point) })
                .Where(p => p.dist < options.matchingOptions.featurePairDistanceThreshold)
                .Select(p => p.pair)
                .ToList();

            options.allFeaturePairsAction(featurePairs);

            var optimalPairs = SimpleRansac.RansacMatch(featurePairs, options.ransacOptions, rand);

            if (optimalPairs.Item2 != null)
            {
                options.optimalFeaturePairsAction(optimalPairs.Item2);
            }
            return optimalPairs.Item1;
        }*/

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

            
            ImageFeatureMatching.NNMatchBruteForce(
                fastFeatures[0].briefDesc,
                fastFeatures[1].briefDesc,
                ImageFeatureMatching.HammingDist,
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


        public static void ExtractFastFeatures(string sourceImage)
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
