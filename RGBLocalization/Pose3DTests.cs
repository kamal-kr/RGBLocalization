using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.Distributions;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RGBLocalization.Utility;
using System.IO;

namespace RGBLocalization
{
    public class Pose3DTests
    {

        public static void TestPosePrimitives()
        {
            DenseVector poseQuat = new DenseVector(new double[] {1, 0, 0, 0});
            DenseVector posePosition = new DenseVector(new double[] {0, 0, 1});
            DenseMatrix inverseCalibration = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240).Inverse();
            DenseMatrix dPixels = new DenseMatrix(3, 1);
            dPixels[0, 0] = 320;
            dPixels[1, 0] = 240;
            dPixels[2, 0] = 1;


            dPixels = (DenseMatrix)dPixels.Append(dPixels);


            Console.WriteLine("New: Using calibration Matrix");
            Console.WriteLine(Pose3D.DPixelToWorld(poseQuat, posePosition, inverseCalibration, dPixels, new DenseMatrix(1, dPixels.ColumnCount, 1.0)).ToString("0.0"));
            //Console.WriteLine("Old: Using calibration Matrix");
            //Console.WriteLine(Pose3D.DPixelToWorld_Old(poseQuat, posePosition, inverseCalibration, dPixels).ToString("0.0"));

            Console.WriteLine("New: No calibration Matrix");
            Console.WriteLine(Pose3D.DPixelToWorld(poseQuat, posePosition, DenseMatrix.Identity(3), dPixels, new DenseMatrix(1, dPixels.ColumnCount, 1.0)).ToString("0.0"));
            //Console.WriteLine("Old: No calibration Matrix");
            //Console.WriteLine(Pose3D.DPixelToWorld_Old(poseQuat, posePosition, DenseMatrix.Identity(4), dPixels).ToString("0.0"));
        }

        //confirm 3d geometry to infer the pose given many world points and 2d pixels
        public static void TestPoseEstimation()
        {
            DenseVector poseQuat = new DenseVector(new double[] { 1, 0, 0, 0 });
            DenseVector posePosition = new DenseVector(new double[] { 0, 0, 1 });

            
            DenseMatrix calibrationMatrix = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240);
            
            int numPoints = 100;

            DenseMatrix scalesToInfer = DenseMatrix.CreateRandom(1, numPoints, new MathNet.Numerics.Distributions.ContinuousUniform(0, 100));
            //DenseMatrix scalesToInfer = new DenseMatrix(1, numPoints, 1);
            //scalesToInfer[0, 0] = 2;
            DenseMatrix scaleInvPoints = DenseMatrix.CreateRandom(2, numPoints, new MathNet.Numerics.Distributions.ContinuousUniform(0, 1.0));
            DenseMatrix rtToInfer = DenseMatrix.Identity(4);
            rtToInfer = DenseMatrix.CreateRandom(4, 4, new MathNet.Numerics.Distributions.ContinuousUniform(0, 1.0));
            //rtToInfer.ClearRow(3);
            //rtToInfer[3, 3] = 1;
            //rtToInfer[0, 2] = 0.5;


            DenseVector rowOfOnes = new DenseVector(numPoints, 1.0);
            DenseMatrix knownWorldPoints = (DenseMatrix)
                                            rtToInfer.Multiply(scalesToInfer
                                                                    .Replicate(3, 1)
                                                                    .PointwiseMultiply(scaleInvPoints.InsertRow(2, rowOfOnes))
                                                                    .InsertRow(3, rowOfOnes));

            List<double> errorPlot = new List<double>();

            Pose3D.InferRTFromScaleInvariantWorld_doesntwork(
                                                    (DenseMatrix)knownWorldPoints.SubMatrix(0, 3, 0, numPoints),
                                                    scaleInvPoints,
                                                    new Pose3D.GradientDescentOptions<Tuple<DenseMatrix, DenseMatrix>>
                                                        {
                                                            iterationListener = (i, errorMatrix, parameters) =>
                                                                {
                                                                    errorPlot.Add(errorMatrix.PointwiseMultiply(errorMatrix).SumRows().Transpose().SumRows().ApplyFunction(Math.Sqrt)[0, 0]);
                                                                    //Console.WriteLine("Iteration: {0}\tErr: {1}", i, );
                                                                    Console.WriteLine("Error:");
                                                                    Console.WriteLine("{0: 0.0}", errorMatrix);
                                                                    Console.WriteLine("Inferred R|T:");
                                                                    Console.WriteLine("{0: 0.00}", parameters.Item1);
                                                                    Console.WriteLine("Real R|T:");
                                                                    Console.WriteLine("{0: 0.00}", rtToInfer);
                                                                    Console.WriteLine("Inferred Scale:");
                                                                    Console.WriteLine("{0: 0.00}", parameters.Item2);
                                                                    Console.WriteLine("Real Scale:");
                                                                    Console.WriteLine("{0: 0.00}", scalesToInfer);
                                                                },
                                                            errorTolerance = 0.0001,
                                                            numIterations = 2000,
                                                            learningRate = 0.00001
                                                        });

            File.WriteAllLines(@"C:\Kamal\RSE\WorkingDirs\BundleAdj\err.txt", errorPlot.Select(d => d.ToString()).ToArray());
        }

        //confirm 3d geometry to infer the pose given many world points and 2d pixels
        public static void TestPoseEstimationOpenCV()
        {
            DenseVector poseQuat = new DenseVector(new double[] { 1, 0, 0, 0 });
            DenseVector posePosition = new DenseVector(new double[] { 0, 0, 0 });

            DenseMatrix calibrationMatrix = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240);
            
            int numPoints = 7;

            Matrix<double> worldPoints = DenseMatrix.CreateRandom(2, numPoints, new ContinuousUniform(100, 300));
            worldPoints = worldPoints.InsertRow(2, DenseVector.CreateRandom(numPoints, new ContinuousUniform(500, 2000)));

            var extParameters = new Emgu.CV.ExtrinsicCameraParameters();
            extParameters.RotationVector = new Emgu.CV.RotationVector3D(new double[] { 4, 5, 6 });
            extParameters.TranslationVector = new Emgu.CV.Matrix<double>(new double[] { 1, 2, 3 });
            
            Console.WriteLine("Known extrinsic:\n {0}", RGBLocalization.Utility.MatrixExtensions.ToString(extParameters.ExtrinsicMatrix));
            Console.WriteLine("Known extrinsic (translation vector):\n {0}", RGBLocalization.Utility.MatrixExtensions.ToString(extParameters.TranslationVector));

            var intParameters = new Emgu.CV.IntrinsicCameraParameters();
            intParameters.IntrinsicMatrix = calibrationMatrix.ToEmguMatrix();

            var projectedPoints = 
                Emgu.CV.CameraCalibration.ProjectPoints(
                worldPoints.ColumnEnumerator().Select(col => new MCvPoint3D32f((float)col.Item2[0], (float)col.Item2[1], (float)col.Item2[2])).ToArray(),
                extParameters,
                intParameters);

            Console.WriteLine("Known world: \n{0:0.00}", worldPoints);
            Console.WriteLine("Projeted points: \n{0:0.00}", projectedPoints.ToMatrix(p => new double[] { p.X, p.Y }, 2).Transpose());

            var inferredExtParameters =
            Emgu.CV.CameraCalibration.FindExtrinsicCameraParams2(
                worldPoints.ColumnEnumerator().Select(col => new MCvPoint3D32f((float)col.Item2[0], (float)col.Item2[1], (float)col.Item2[2])).ToArray(),
                projectedPoints,
                intParameters
                );

            Console.WriteLine("Inferred Ext: \n{0:0.00}", RGBLocalization.Utility.MatrixExtensions.ToString(inferredExtParameters.ExtrinsicMatrix));
            Console.WriteLine("Inferred Ext (translation vector): \n{0:0.00}", RGBLocalization.Utility.MatrixExtensions.ToString(inferredExtParameters.TranslationVector));
        }

        public static void TestPoseEstOpenCVRealData()
        {
            string mapFile = @"C:\Kamal\RSE\WorkingDirs\Visualizaton\espresso-1.bag.dump.map";
            string poseFile = @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt";

            var intParameters = new Emgu.CV.IntrinsicCameraParameters();
            intParameters.IntrinsicMatrix = ((DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240)).ToEmguMatrix();

            //var identityExtParameters = new Emgu.CV.ExtrinsicCameraParameters();
            //identityExtParameters.RotationVector = new Emgu.CV.RotationVector3D(new double[] { 0, 0, 0 });
            //identityExtParameters.TranslationVector = new Emgu.CV.Matrix<double>(new double[] { 0, 0, 0 });

            var poseData = ImageMap.ParsePoseData(File.ReadLines(poseFile),
                                    (frameID, poseQuaternion, posePosition) =>
                                        new
                                        {
                                            frameID,
                                            poseQuaternion,
                                            posePosition
                                        })
                                        .ToDictionary(p => p.frameID, p => new { p.posePosition, p.poseQuaternion });
            
            var imageMap = 
                ImageMap.LoadImageMap(mapFile, (frameID, projectedPt, point3D, descriptor) =>
                                new
                                {
                                    frameID,
                                    projectedPoint = projectedPt,
                                    //openCVProjectedPoint = Emgu.CV.CameraCalibration.ProjectPoints(
                                    //                Enumerable.Repeat(new MCvPoint3D32f((float)point3D[0], (float)point3D[1], (float)point3D[2]), 1).ToArray(),
                                    //                identityExtParameters,
                                    //                intParameters)[0],
                                    point3D,
                                    descriptor
                                })
                                .ToList();

            var realAndInferred =
                imageMap.GroupBy(i => i.frameID)
                .Select(f =>
                            new
                            {
                                frameID = f.Key,
                                projectedPoints = f.Select(ff => ff.projectedPoint),
                                //openCVProjectedPoints = f.Select(ff => ff.openCVProjectedPoint),
                                points3D = f.Select(ff => ff.point3D),
                                poseData[f.Key].posePosition,
                                poseData[f.Key].poseQuaternion,
                                inferredPositionVector =
                                Emgu.CV.CameraCalibration.FindExtrinsicCameraParams2(
                                    f.Select(col => new MCvPoint3D32f((float)col.point3D[0], (float)col.point3D[1], (float)col.point3D[2])).ToArray(),
                                    f.Select(ff => ff.projectedPoint).ToArray(),
                                    intParameters
                                    )
                                    .TranslationVector
                            })
                    .ToArray();

            var errors =
                realAndInferred
                .Select(ri => new DenseMatrix(ri.inferredPositionVector.Data).Subtract(ri.posePosition.ToColumnMatrix()).L2Norm())
                .Select(d => d.ToString());

            var lengths =
            realAndInferred
                .Select(ri => ri.posePosition.ToColumnMatrix().Magnitude())
                .Select(d => d.ToString());
            
            File.WriteAllLines(@"C:\Kamal\RSE\WorkingDirs\Octave\data\err.txt", errors);
            File.WriteAllLines(@"C:\Kamal\RSE\WorkingDirs\Octave\data\len.txt", lengths);

            var worstErr =
                realAndInferred
                .OrderByDescending(ri => new DenseMatrix(ri.inferredPositionVector.Data).Subtract(ri.posePosition.ToColumnMatrix()).L2Norm());


            foreach (var ri in worstErr.Take(10))
            {
                Console.WriteLine(ri.frameID);
                Console.WriteLine("Numpoints:{0}", ri.projectedPoints.Count());
                Console.WriteLine("Real: {0: 0.00}", ri.posePosition);
                //                Console.WriteLine("Real (quat): {0}", ri.poseQuaternion);
                Console.WriteLine("Inferred: {0: 0.00}", RGBLocalization.Utility.MatrixExtensions.ToString(ri.inferredPositionVector));
                //                Console.WriteLine("ProjectedPointsfromfile:\n {0:0.00}", ri.projectedPoints.ToMatrix(p => new double[] { p.X, p.Y }, 2));
                //Console.WriteLine("OpenCV:\n {0:0.00}", ri.openCVProjectedPoints.ToMatrix(p => new double[] { p.X, p.Y }, 2));
            }

        }


        public static void TestOpenCVProjection()
        {
            int numPoints = 1;

            Matrix<double> worldPoints = DenseMatrix.CreateRandom(2, numPoints, new ContinuousUniform(100, 300));
            worldPoints = worldPoints.InsertRow(2, DenseVector.CreateRandom(numPoints, new ContinuousUniform(500, 2000)));

            var extParameters = new Emgu.CV.ExtrinsicCameraParameters();
            extParameters.RotationVector = new Emgu.CV.RotationVector3D(new double[] { 0, 0, 0 });
            extParameters.TranslationVector = new Emgu.CV.Matrix<double>(new double[] { 1, 2, 3 });

            DenseMatrix calibrationMatrix = (DenseMatrix)Pose3D.CreateCalibrationMatrix(525, 320, 240);

            var intParameters = new Emgu.CV.IntrinsicCameraParameters();
            intParameters.IntrinsicMatrix = calibrationMatrix.ToEmguMatrix();

            var openCVProjectedPoints =
                Emgu.CV.CameraCalibration.ProjectPoints(
                worldPoints.ColumnEnumerator().Select(col => new MCvPoint3D32f((float)col.Item2[0], (float)col.Item2[1], (float)col.Item2[2])).ToArray(),
                extParameters,
                intParameters);

            Console.WriteLine("Open CV Projeted points: \n{0:0.00}", openCVProjectedPoints.ToMatrix(p => new double[] { p.X, p.Y }, 2).Transpose());

            var myProjectedPoints = Pose3D.WorldToImagePoints(
                                            worldPoints, 
                                            calibrationMatrix,
                                            new DenseVector(new double[] { 1, 2, 3 }), 
                                            new DenseVector(new double[] { 1, 0, 0, 0 })
                                            .QuaternionToRotation());
            Console.WriteLine("My Projeted points: \n{0:0.00}", myProjectedPoints);
        }
    }

}
