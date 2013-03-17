using Emgu.CV.Structure;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RGBLocalization
{
    public class ImageMap
    {
        public static IEnumerable<T> ParsePoseData<T>(IEnumerable<string> poseLines, Func<string, DenseVector, DenseVector, T> map)
        {
            return
                poseLines
                .Select(l => l.Split(new char[] { ' ', ':' }, StringSplitOptions.RemoveEmptyEntries))
                .Select(l => 
                                map(l[0] + "-" + l[1], 
                                new DenseVector(l.Skip(2).Take(4).Select(f => Double.Parse(f)).ToArray()),
                                new DenseVector(l.Skip(6).Take(3).Select(f => Double.Parse(f)).ToArray())));
        }

        static Func<string, string> FrameID = (imageFileName) =>
            Path.GetFileNameWithoutExtension(imageFileName)
                    .Replace("-image", "");

        public static IEnumerable<double[]> GetRGBDFeaturePoints(string imageFile, string depthFile, RGBMatch.FeatureExtractionOptions options)
        {
            //to confirm if this is the right way of reading depth values
            var depthInfo = new Emgu.CV.Image<Gray, double>(depthFile).Data;

            return
           RGBMatch.FastFeatureExtRaw(
                           new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(imageFile),
                           options)
               .Select(keyPoint => new double[]
                            {
                                 keyPoint.Point.X,
                                 keyPoint.Point.Y,
                                //yet to confirm if this is the right wa of reading the depth in meters.
                                //the original image is a 16 bit single channel png with depth values in mm
                                //i'm guessing that highest intensity corresponds to black (which should be 0 depth)
                                ((double)depthInfo[(int) keyPoint.Point.Y, (int)keyPoint.Point.X, 0])/1000.0
                            })
                .Select(d =>
                            {
                               // Console.WriteLine("[{0},{1}] -> {2}", (int)d[0], (int)d[1],  depthInfo[(int)d[1], (int)d[0], 0]);//depthInfo[(int)d[0], (int)d[1]]);
                                return d;
                            });
        }

        public static IEnumerable<T> CreateImageMap<T>(
            IEnumerable<Tuple<string, string>> imageAndDepthFileNames,
            string poseTextFile, 
            RGBMatch.FeatureExtractionOptions options,
            DenseMatrix calibrationMatrix,
            Func<string, DenseMatrix, T> map)
        {
            var framePoses = ParsePoseData(
                                File.ReadLines(poseTextFile),
                                (frameID, poseQuaternion, posePosition) => new
                                {
                                    frameID,
                                    poseQuaternion,
                                    posePosition
                                })
                                .ToDictionary(p => p.frameID, p=> new { p.poseQuaternion, p.posePosition});

            DenseMatrix inverseCalibration = (DenseMatrix)calibrationMatrix.Inverse();

            return
            imageAndDepthFileNames
                .Select(f => new { imageFileName = f.Item1, depthFileName = f.Item2 })
                .Select(rgbdPair => new
                                    {
                                        frameId = FrameID(rgbdPair.imageFileName),
                                        rgbPoints = GetRGBDFeaturePoints(rgbdPair.imageFileName, rgbdPair.depthFileName, options)
                                                        //filter out depth 0 points
                                                        .Where(d => d[2] != 0)
                                                        .Select(d => d.Concat(Enumerable.Repeat(1.0,1)))
                                                        .SelectMany(d => d)
                                                        .ToArray()
                                    })
                .Where(im => im.rgbPoints.Length > 0)
                .Select(r => new 
                                {
                                    r.frameId,
                                    //dennsematrix constructor expects the array to be column wise
                                    dPixels = new DenseMatrix(4, r.rgbPoints.Length/4, r.rgbPoints)
                                })
                .Select(p => new 
                                {
                                    p.frameId,
                                    worldPoints = 
                                        Pose3D.DPixelToWorld(
                                            framePoses[p.frameId].poseQuaternion,
                                            framePoses[p.frameId].posePosition,
                                            inverseCalibration,
                                            p.dPixels)
                                }
                        )
                .Select(p => map(p.frameId, p.worldPoints));
        }
    }
}
