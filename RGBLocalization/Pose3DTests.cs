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

            Pose3D.InferRTFromScaleInvariantWorld(
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
            calibrationMatrix = DenseMatrix.Identity(3);
            
            int numPoints = 7;

            Matrix<double> worldPoints = DenseMatrix.CreateRandom(2, numPoints, new ContinuousUniform(100, 300));
            worldPoints = worldPoints.InsertRow(2, DenseVector.CreateRandom(numPoints, new ContinuousUniform(500, 2000)));

            //var imagePoints = worldPoints.WorldToImagePoints(calibrationMatrix, posePosition, poseQuat.QuaternionToRotation());


            var extParameters = new Emgu.CV.ExtrinsicCameraParameters();
            extParameters.RotationVector = new Emgu.CV.RotationVector3D(new double[] { 4, 5, 6 });
            extParameters.TranslationVector = new Emgu.CV.Matrix<double>(new double[] { 1, 2, 3 });
            
            Console.WriteLine("Known extrinsic:\n {0}", RGBLocalization.Utility.MatrixExtensions.ToString(extParameters.ExtrinsicMatrix));

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

        }
    }

}
