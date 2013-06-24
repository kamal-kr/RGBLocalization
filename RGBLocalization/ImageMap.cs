using Emgu.CV.Structure;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;
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

        public static IEnumerable<T> GetRGBDFeaturePoints<T>(
            string imageFile, 
            string depthFile, 
            ImageFeatureExtraction.FeatureExtractionOptions options, 
            Func<double, double, double, MKeyPoint, Emgu.CV.Image<Gray, byte>, T> outputMap)
        {
            var depthInfo = new Emgu.CV.Image<Gray, double>(depthFile).Data;
            var image = new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>(imageFile);
            return options.DoExtract(image)
                    .Select(keyPoint => outputMap(
                                        keyPoint.Point.X, 
                                        keyPoint.Point.Y,
                                        //the original image is a 16 bit single channel png with depth values in mm
                                        //the highest intensity corresponds to black (which should be 0 depth)
                                        ((double)depthInfo[(int) keyPoint.Point.Y, (int)keyPoint.Point.X, 0])/1000.0,
                                        keyPoint,
                                        image));
        }

        public static IEnumerable<T> CreateImageMap<T>(
            IEnumerable<Tuple<string, string>> imageAndDepthFileNames,
            string poseTextFile, 
            ImageFeatureExtraction.FeatureExtractionOptions options,
            DenseMatrix calibrationMatrix,
            Func<string, DenseMatrix, Emgu.CV.Matrix<byte>[], DenseMatrix, DenseMatrix, T> map)
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
                                        rgbdPoints = GetRGBDFeaturePoints(rgbdPair.imageFileName, rgbdPair.depthFileName, options,
                                                                        (x, y, depth, keypoint, image) =>
                                                                                new
                                                                                {
                                                                                    x,
                                                                                    y,
                                                                                    depth,
                                                                                    featureDescriptor = ImageFeatureExtraction.ExtractBriefFeatureDescriptors(image, keypoint)
                                                                                })
                                                        //filter out depth 0 points and points where we cannot get feature descriptors
                                                        .Where(dPixel => dPixel.depth != 0 && dPixel.featureDescriptor != null)
                                                        .ToArray(),
                                    })
                .Where(im => im.rgbdPoints.Length > 0)
                .Select(r => new
                                {
                                    r.frameId,
                                    //dennsematrix constructor expects the array to be column wise
                                    homogeneousPixels = new DenseMatrix(3,
                                                                        r.rgbdPoints.Length,
                                                                        r.rgbdPoints.SelectMany(p => new double[] {p.x, p.y, 1.0}).ToArray()),
                                    depths = new DenseMatrix(1, r.rgbdPoints.Length, r.rgbdPoints.Select(p => p.depth).ToArray()),
                                    featureDescriptors = r.rgbdPoints.Select(fd => fd.featureDescriptor).ToArray()
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
                                    p.featureDescriptors,
                                    p.homogeneousPixels,
                                    p.depths
                                }
                        )
                .Select(p => map(p.frameId, p.worldPoints, p.featureDescriptors, p.homogeneousPixels, p.depths));
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
                                    (frameId, worldPoints, featureDesc, imagePoints, depths) => new { frameId, worldPoints, featureDesc, imagePoints, depths })
                          .SelectMany(w => w.worldPoints.ColumnEnumerator()
                                            .Select(c => new { 
                                                                worldPoint = c.Item2, 
                                                                w.frameId, 
                                                                featureDesc = w.featureDesc[c.Item1],
                                                                imagePoint = new System.Drawing.PointF((float)w.imagePoints[0, c.Item1], (float)w.imagePoints[1, c.Item1]),
                                                                depth = w.depths[0, c.Item1]
                                                            }))
                          .ToList();

            Func<Emgu.CV.Matrix<byte>, string> rowMatrixToTSV = m => String.Join("\t", Enumerable.Range(0, m.Size.Width).Select(i => m[0, i].ToString()));

            File.WriteAllLines(Path.Combine(Path.GetDirectoryName(outputMapFile), Path.GetFileNameWithoutExtension(outputMapFile) + ".asc"),
                                imageMap.Select(p => String.Format("{0},{1},{2}", p.worldPoint[0], p.worldPoint[1], p.worldPoint[2])));

            File.WriteAllLines(outputMapFile,
                                imageMap.Select(p => String.Format("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}", 
                                                    p.frameId,
                                                    p.imagePoint.X,
                                                    p.imagePoint.Y,
                                                    p.depth,
                                                    p.worldPoint[0], 
                                                    p.worldPoint[1], 
                                                    p.worldPoint[2],
                                                    rowMatrixToTSV(p.featureDesc))));
        }

        public static IEnumerable<T> LoadImageMap<T>(string mapFileName, Func<string, System.Drawing.PointF, double, DenseVector, byte[], T> outputMap)
        {
            return
            File.ReadLines(mapFileName)
                .Select(l => l.Split('\t'))
                .Select(l => new
                {
                    frameId = l[0],
                    imagePoint = new System.Drawing.PointF(Single.Parse(l[1]), Single.Parse(l[2])),
                    depth = Double.Parse(l[3]),
                    point3D = new DenseVector(l.Skip(4).Take(3).Select(s => Double.Parse(s)).ToArray()),
                    descriptor = l.Skip(7).Take(32).Select(s => Byte.Parse(s)).ToArray()
                })
                .Select(l => outputMap(l.frameId, l.imagePoint, l.depth, l.point3D, l.descriptor));
        }


    }
}
