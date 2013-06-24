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
    public class CamLocalization
    {
        public static string MatchRGBImage(
            string imageMapFile, 
            string imageFile, 
            ImageFeatureExtraction.FeatureExtractionOptions options,
            Random random)
        {
            var imageMap = ImageMap.LoadImageMap(imageMapFile, (frameID, imagePoint, depth, pt3D, descriptor) =>
                                            new
                                            {
                                                frameID,
                                                point3D = new MCvPoint3D32f((float)pt3D[0], (float)pt3D[1], (float)pt3D[2]),
                                                descriptor
                                            })
                                            .Select((p, lineNumber) => new { p, lineNumber });

            var frameWiseMap =
                imageMap
                .GroupBy(i => i.p.frameID)
                .ToDictionary(frame => frame.Key,
                                frame => new { 
                                        point3D = frame.OrderBy(f => f.lineNumber).Select(f => f.p.point3D).ToArray(),
                                        descMatrx = frame.OrderBy(f => f.lineNumber).Select(f => f.p.descriptor).ToEmguMatrix(b => b)
                                    });
                

            var image = new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(imageFile);
            var imageFeaturePoints =
                options.DoExtract(image)
                .Select(kp =>
                                new
                                {
                                    featureDescriptor = ImageFeatureExtraction.ExtractBriefFeatureDescriptors(image, kp),
                                    kp
                                })
                .Where(kp => kp.featureDescriptor != null)
                .ToArray();

          var imageFeatureDesc = imageFeaturePoints.ToEmguMatrix(p => p.featureDescriptor.EnumerateRowwise().ToArray());
          
          var matchingOptions = new ImageFeatureMatching.FeatureMatchingOptions();

          Func<string, IEnumerable<Tuple<MCvPoint3D32f, MKeyPoint>>> MatchFeatures =
              (frame) => ImageFeatureMatching.NNMatchBruteForce(imageFeatureDesc, frameWiseMap[frame].descMatrx, matchingOptions.distanceFunction, matchingOptions.featurePairDistanceThreshold,
                            (i1, i2, dist) => new { i1, i2, dist })
                            .Where(p => p.dist < matchingOptions.featurePairDistanceThreshold)
                            .Select(p => new Tuple<MCvPoint3D32f, MKeyPoint>(frameWiseMap[frame].point3D[p.i2], imageFeaturePoints[p.i1].kp))
                            .ToArray();

          DenseMatrix camCalibration = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240);
          var intParameters = new Emgu.CV.IntrinsicCameraParameters();
          intParameters.IntrinsicMatrix = camCalibration.ToEmguMatrix();

            Func<Emgu.CV.ExtrinsicCameraParameters, Tuple<MCvPoint3D32f, MKeyPoint>, double> modelEvaluator =
                            (extParam, featurePair) =>
                                Emgu.CV.CameraCalibration.ProjectPoints(new MCvPoint3D32f[] { featurePair.Item1 }, extParam, intParameters)[0]
                                .distanceTo(featurePair.Item2.Point);

            Func<IEnumerable<Tuple<MCvPoint3D32f, MKeyPoint>>, Tuple<Emgu.CV.ExtrinsicCameraParameters, double>> modelFitter =
                matchedFeatures =>
                {
                    var model = Emgu.CV.CameraCalibration.FindExtrinsicCameraParams2(matchedFeatures.Select(m => m.Item1).ToArray(), matchedFeatures.Select(m => m.Item2.Point).ToArray(), intParameters);
                    return new Tuple<ExtrinsicCameraParameters, double>(model,
                                    matchedFeatures.Average(fp => Math.Pow(modelEvaluator(model, fp), 2)));
                };

            var ransacOptions = new SimpleRansac.RansacOptions 
                                    { 
                                        minNumInliers = 15,
                                        numMinSamples = 6,
                                        numTrials = 5,
                                        rand = random,
                                        sqInlierErrorThreshold = 9 //3 pixels
                                    };

            var matchingFrame =
            frameWiseMap
                .Skip(212)
                .AsParallel()
                .Select(kvp => new
                {
                    frame = kvp.Key,
                    featurePairs = MatchFeatures(kvp.Key)
                })
                .Where(k => k.featurePairs.Count() >= ransacOptions.minNumInliers)
                .Select(f => new
                            {
                                f.frame,
                                modelAndError = SimpleRansac.Ransac(f.featurePairs, modelFitter, modelEvaluator, ransacOptions)
                            })
                .ShowProgress(".", 1)
                .Where(f => f.modelAndError.Item1 != null)
                .ShowProgress("!", 1)
                .OrderBy(f => f.modelAndError.Item2)
                .Select(f => String.Format("frame:{0}\terror={1}", f.frame, f.modelAndError.Item2));

            File.WriteAllLines(@Path.Combine(@"C:\Kamal\RSE\TestResults\FrameQuery", Path.GetFileNameWithoutExtension(imageFile)), matchingFrame);

            return "";
        }
    }
}
