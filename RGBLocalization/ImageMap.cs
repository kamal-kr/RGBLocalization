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

        public static IEnumerable<Tuple<double[], double>> GetRGBDFeaturePoints(string imageFile, string depthFile, RGBMatch.FeatureExtractionOptions options)
        {
            //to confirm if this is the right way of reading depth values
            var depthInfo = new Emgu.CV.Image<Gray, double>(depthFile).Data;

            return
           RGBMatch.FastFeatureExtRaw(
                           new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(imageFile),
                           options)
               .Select(keyPoint => new Tuple<double[], double>(
                                 new double[] { keyPoint.Point.X, keyPoint.Point.Y },
                                //the original image is a 16 bit single channel png with depth values in mm
                                //i'm guessing that highest intensity corresponds to black (which should be 0 depth)
                                ((double)depthInfo[(int) keyPoint.Point.Y, (int)keyPoint.Point.X, 0])/1000.0));
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
                                                        .Where(dPixel => dPixel.Item2 != 0)
                                                        .ToArray()
                                    })
                .Where(im => im.rgbPoints.Length > 0)
                .Select(r => new 
                                {
                                    r.frameId,
                                    //dennsematrix constructor expects the array to be column wise
                                    homogeneousPixels = new DenseMatrix(3, 
                                                                        r.rgbPoints.Length, 
                                                                        r.rgbPoints.SelectMany(p => p.Item1.Concat(new double[]{1.0})).ToArray()),
                                    depths = new DenseMatrix(1, r.rgbPoints.Length, r.rgbPoints.Select(p => p.Item2).ToArray())
                                })
                .Select(p => new 
                                {
                                    p.frameId,
                                    worldPoints = 
                                        Pose3D.DPixelToWorld(
                                            framePoses[p.frameId].poseQuaternion,
                                            framePoses[p.frameId].posePosition,
                                            inverseCalibration,
                                            p.homogeneousPixels,
                                            p.depths)
                                }
                        )
                .Select(p => map(p.frameId, p.worldPoints));
        }
    }
}
