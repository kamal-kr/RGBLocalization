using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RGBLocalization
{
    public static class Pose3D
    {
        public static DenseMatrix DPixelToWorld(
                                                    DenseVector poseQuaternion, 
                                                    DenseVector posePosition, 
                                                    DenseMatrix inverseCalibration, 
                                                    DenseMatrix homogeneousPixels,
                                                    DenseMatrix depths)
        {
            return  ((DenseMatrix)
                    poseQuaternion.QuaternionToRotation()
                    .Multiply(DPixelToWorld(inverseCalibration, homogeneousPixels, depths)))
                    .Translate(posePosition);
        }

        public static DenseMatrix Translate(this DenseMatrix worldPoints, DenseVector posePosition)
        {
            return (DenseMatrix)
                new DenseMatrix(posePosition.Count, 1, posePosition.ToArray())
                .Multiply(new DenseMatrix(1, worldPoints.ColumnCount, 1))
                .Add(worldPoints);
        }

        public static DenseMatrix DPixelToWorld(DenseMatrix inverseCalibration, DenseMatrix homogeneousPixels, DenseMatrix depths)
        {
            return (DenseMatrix)
                    new DenseMatrix(3, 1, 1.0)
                    .Multiply(depths)
                    .PointwiseMultiply(inverseCalibration.Multiply(homogeneousPixels));
        }

        public static DenseMatrix QuaternionToRotation(this DenseVector quaternion)
        {
            //http://szeliski.org/Book/: page 44
            
            DenseMatrix m = DenseMatrix.Identity(3);
            var w = quaternion[0];
            var x = quaternion[1];
            var y = quaternion[2];
            var z = quaternion[3];

            m[0, 0] = 1.0 - 2.0 * (y * y + z * z);
            m[0, 1] = 2.0 * (x * y - z * w);
            m[0, 2] = 2.0 * (x * z + y * w);
            m[1, 0] = 2.0 * (x * y + z * w);
            m[1, 1] = 1.0 - 2.0 * (x * x + z * z);
            m[1, 2] = 2.0 * (y * z - x * w);
            m[2, 0] = 2.0 * (x * z - y * w);
            m[2, 1] = 2.0 * (y * z + x * w);
            m[2, 2] = 1.0 - 2.0 * (x * x + y * y);

            return m;
        }

        public static DenseMatrix CreateCalibrationMatrix(double focalLength, double centerX, double centerY)
        {
            DenseMatrix cal = DenseMatrix.Identity(3);
            cal[0, 0] = focalLength;
            cal[0, 2] = centerX;
            cal[1, 1] = focalLength;
            cal[1, 2] = centerY;
            return cal;
        }
    }
}
