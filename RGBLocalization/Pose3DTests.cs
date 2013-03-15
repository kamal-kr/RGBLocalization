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
            DenseMatrix dPixels = new DenseMatrix(4, 1);
            dPixels[0, 0] = 320;
            dPixels[1, 0] = 240;
            dPixels[2, 0] = 1;
            dPixels[3, 0] = 1;

            dPixels = (DenseMatrix)dPixels.Append(dPixels);

            var worldC = Pose3D.DPixelToWorld(poseQuat, posePosition, inverseCalibration, dPixels);

            Console.WriteLine(worldC.ToString("0.0"));

            worldC = Pose3D.DPixelToWorld(poseQuat, posePosition, DenseMatrix.Identity(4), dPixels);

            Console.WriteLine(worldC.ToString("0.0"));
        }
    }
}
