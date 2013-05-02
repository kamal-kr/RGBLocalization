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

        public static IEnumerable<Tuple<double[], double, MKeyPoint>> GetRGBDFeaturePoints(string imageFile, string depthFile, ImageFeatureExtraction.FeatureExtractionOptions options)
        {
            var depthInfo = new Emgu.CV.Image<Gray, double>(depthFile).Data;

            return options.DoExtract(new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(imageFile))
                    .Select(keyPoint => new Tuple<double[], double, MKeyPoint>(
                                        new double[] { keyPoint.Point.X, keyPoint.Point.Y },
                                    //the original image is a 16 bit single channel png with depth values in mm
                                    //the highest intensity corresponds to black (which should be 0 depth)
                                    ((double)depthInfo[(int) keyPoint.Point.Y, (int)keyPoint.Point.X, 0])/1000.0,
                                    keyPoint));
        }

        public static IEnumerable<T> CreateImageMap<T>(
            IEnumerable<Tuple<string, string>> imageAndDepthFileNames,
            string poseTextFile, 
            ImageFeatureExtraction.FeatureExtractionOptions options,
            DenseMatrix calibrationMatrix,
            Func<string, DenseMatrix, Emgu.CV.Matrix<byte>[], T> map)
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
                .AsParallel()
                .Select(f => new { imageFileName = f.Item1, depthFileName = f.Item2 })
                .Select(rgbdPair => new
                                    {
                                        frameId = FrameID(rgbdPair.imageFileName),
                                        rgbdPoints = GetRGBDFeaturePoints(rgbdPair.imageFileName, rgbdPair.depthFileName, options)
                                            //filter out depth 0 points
                                                        .Where(dPixel => dPixel.Item2 != 0)
                                                        .ToArray(),
                                        rgbdPair.imageFileName
                                    })
                .Where(im => im.rgbdPoints.Length > 0)
                .Select(p => new
                            {
                                p.frameId,
                                p.rgbdPoints,
                                featureDescriptors = ImageFeatureExtraction
                                                        .ExtractBriefFeatureDescriptors(new Emgu.CV.Image<Gray, byte>(p.imageFileName),
                                                                                            p.rgbdPoints.Select(fp => fp.Item3).ToArray())
                            })
                .Select(p => new
                            {
                                p.frameId,
                                featDescPoints = p.rgbdPoints
                                                   .Select((pp, i) => new {pp, i})
                                                   .Join(p.featureDescriptors, o => o.i, featDesc => featDesc.Item1, (rgbdi, featDesc) => new {rgbd = rgbdi.pp, featDesc})
                                                   .OrderBy(fd => fd.featDesc.Item1)
                                                   .Select(fd => new {featureDescriptor = fd.featDesc.Item2, fd.rgbd})
                                                   .ToArray()
                            })
                .Where(p => p.featDescPoints.Length > 0)
                .Select(r => new
                                {
                                    r.frameId,
                                    //dennsematrix constructor expects the array to be column wise
                                    homogeneousPixels = new DenseMatrix(3,
                                                                        r.featDescPoints.Length,
                                                                        r.featDescPoints.SelectMany(p => p.rgbd.Item1.Concat(new double[] { 1.0 })).ToArray()),
                                    depths = new DenseMatrix(1, r.featDescPoints.Length, r.featDescPoints.Select(p => p.rgbd.Item2).ToArray()),
                                    featureDescriptors = r.featDescPoints.Select(fd => fd.featureDescriptor).ToArray()
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
                                            p.depths),
                                    p.featureDescriptors
                                }
                        )
                .Select(p => map(p.frameId, p.worldPoints, p.featureDescriptors));
        }

        public static void SaveImageMap(IEnumerable<Tuple<string, string>> imageAndDepthFiles, 
                                        string poseFile,
                                        string outputMapFile)
        {
            var imageMap =
                ImageMap.CreateImageMap(imageAndDepthFiles,
                                    poseFile,
                                    new ImageFeatureExtraction.FeatureExtractionOptions(),
                                    Pose3D.CreateCalibrationMatrix(525, 320, 240),
                                    (frameId, worldPoints, featureDesc) => new { frameId, worldPoints, featureDesc })
                          .SelectMany(w => w.worldPoints.ColumnEnumerator()
                                            .Select(c => new { worldPoint = c.Item2, w.frameId, featureDesc = w.featureDesc[c.Item1] }))
                          .ToList();

            Func<Emgu.CV.Matrix<byte>, string> rowMatrixToTSV = m => String.Join("\t", Enumerable.Range(0, m.Size.Width).Select(i => m[0, i].ToString()));

            File.WriteAllLines(Path.Combine(Path.GetDirectoryName(outputMapFile), Path.GetFileNameWithoutExtension(outputMapFile) + ".asc"),
                                imageMap.Select(p => String.Format("{0},{1},{2}", p.worldPoint[0], p.worldPoint[1], p.worldPoint[2])));

            File.WriteAllLines(outputMapFile,
                                imageMap.Select(p => String.Format("{0}\t{1}\t{2}\t{3}\t{4}", 
                                                    p.frameId, 
                                                    p.worldPoint[0], 
                                                    p.worldPoint[1], 
                                                    p.worldPoint[2],
                                                    rowMatrixToTSV(p.featureDesc))));
        }

        public static IEnumerable<T> LoadImageMap<T>(string mapFileName, Func<string, DenseVector, DenseVector, T> outputMap)
        {
            return
            File.ReadLines(mapFileName)
                .Select(l => l.Split('\t'))
                .Select(l => new
                {
                    frameId = l[0],
                    point3D = new DenseVector(l.Skip(1).Take(3).Select(s => Double.Parse(s)).ToArray()),
                    descriptor = new DenseVector(l.Skip(4).Take(32).Select(s => Double.Parse(s)).ToArray())
                })
                .Select(l => outputMap(l.frameId, l.point3D, l.descriptor));
        }
    }
}
