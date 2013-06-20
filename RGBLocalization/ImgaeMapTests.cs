using Emgu.CV.Structure;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using RGBLocalization.Utility;
using System.Threading.Tasks;

namespace RGBLocalization
{
    public class ImageMapTests
    {
        public static void TestImageMapConsistency(string mapFile, string poseFile)
        {
            Func<System.Drawing.PointF, System.Drawing.PointF, double> distance2D = (p1, p2) =>
                            Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));

            Func<DenseVector, System.Drawing.PointF> ToPointF = v => new System.Drawing.PointF((float)v[0], (float)v[1]);

            //string mapFile = @"C:\Kamal\RSE\WorkingDirs\Visualizaton\espresso-1.bag.dump.map";
            //string poseFile = @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt";

            DenseMatrix camCalibration = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240);
            var intParameters = new Emgu.CV.IntrinsicCameraParameters();
            intParameters.IntrinsicMatrix = camCalibration.ToEmguMatrix();

            var poseData = ImageMap.ParsePoseData(File.ReadLines(poseFile),
                                    (frameID, poseQuaternion, posePosition) =>
                                        new
                                        {
                                            frameID,
                                            poseQuaternion,
                                            posePosition
                                        })
                                        .ToDictionary(p => p.frameID, p => new { p.posePosition, p.poseQuaternion });

            Func<DenseVector, DenseVector, DenseVector, System.Drawing.PointF> OpenCVProject =
                (point3D, posePosition, poseQuat) =>
                {
                    var extParameters = new Emgu.CV.ExtrinsicCameraParameters();
                    extParameters.RotationVector = new Emgu.CV.RotationVector3D();
                    extParameters.RotationVector.RotationMatrix = poseQuat.QuaternionToRotation().Inverse().ToEmguMatrix();
                    extParameters.TranslationVector = (-posePosition).ToColumnMatrix().ToEmguMatrix();

                    return
                    Emgu.CV.CameraCalibration.ProjectPoints(
                    new MCvPoint3D32f[] { new MCvPoint3D32f((float)point3D[0], (float)point3D[1], (float)point3D[2]) },
                    extParameters,
                    intParameters)[0];
                };

            var imageMap =
                ImageMap.LoadImageMap(mapFile, (frameID, imagePoint, depth, point3D, descriptor) =>
                                new
                                {
                                    frameID,
                                    projectedPoint = OpenCVProject(point3D, poseData[frameID].posePosition, poseData[frameID].poseQuaternion),
                                    myImpProjectedPoint = (DenseVector)Pose3D.WorldToImagePoints(point3D.ToColumnMatrix(), camCalibration, poseData[frameID].posePosition, poseData[frameID].poseQuaternion.QuaternionToRotation()).Column(0),
                                    imagePoint,
                                    depth,
                                    point3D,
                                    descriptor
                                }).ToList();

            Console.WriteLine("OpevCV vs. myImp Projection error: {0}", imageMap.Average(p => distance2D(ToPointF(p.myImpProjectedPoint), p.projectedPoint)));
            Console.WriteLine("OpevCV vs. imagepoints Projection error: {0}", imageMap.Average(p => distance2D(p.imagePoint, p.projectedPoint)));

            foreach (var p in imageMap.Where(p => distance2D(ToPointF(p.myImpProjectedPoint), p.projectedPoint) > 1000).Take(10))
            {
                Console.WriteLine("----------");
                Console.WriteLine("OpenCV:\t{0}", p.projectedPoint);
                Console.WriteLine("Image:\t{0}", p.imagePoint);
                Console.WriteLine("Depth:\t{0}", p.depth);
                Console.WriteLine("point3d:\t {0}", p.point3D);
                Console.WriteLine("pose:\t {0}", poseData[p.frameID].posePosition);
                Console.WriteLine("quat:\t {0}", poseData[p.frameID].poseQuaternion);
                Console.WriteLine("my proj impl:\t{0}",
                    p.point3D.ToColumnMatrix().WorldToImagePoints(camCalibration, poseData[p.frameID].posePosition, poseData[p.frameID].poseQuaternion.QuaternionToRotation()));

                Console.WriteLine("Re-inferred 3d point: {0}",
                    Pose3D.DPixelToWorld(
                        poseData[p.frameID].poseQuaternion,
                        poseData[p.frameID].posePosition,
                        (DenseMatrix)camCalibration.Inverse(),
                        new DenseMatrix(3, 1, new double[] { p.imagePoint.X, p.imagePoint.Y, 1.0 }),
                        new DenseMatrix(1, 1, new double[] { p.depth })));
            }
        }

        public static void TestImageMap()
        {
            string streamDirName = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump";
            string poseFile = @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt";
            string workingDir = @"C:\Kamal\RSE\WorkingDirs\Visualizaton";

            //1302389028-661940549
            var testPairs = Directory.GetFiles(streamDirName, "*-image.png")
                            .Select(f => new Tuple<string, string>(f, f.Replace("image", "depth")));
            
            ImageMap.SaveImageMap(testPairs, poseFile, Path.Combine(workingDir, Path.GetFileName(streamDirName)+".map"));
        }
    }
}