using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            
            int numPoints = 10;

            DenseMatrix dPixels = DenseMatrix.CreateRandom(4, numPoints, new MathNet.Numerics.Distributions.ContinuousUniform(0, 100));
            dPixels.SetRow(4, Enumerable.Repeat(1.0, numPoints).ToArray());

            //var worldC = Pose3D.DPixelToWorld(poseQuat, posePosition, (DenseMatrix)calibrationMatrix.Inverse(), dPixels);


        }
    }
}
