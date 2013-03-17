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
        public static DenseMatrix DPixelToWorld(DenseVector poseQuaternion, DenseVector posePosition, DenseMatrix inverseCalibration,DenseMatrix dPixels)
        {
            return (DenseMatrix)
                        posePosition.PositionToInverseTranslation()
                        .Multiply(poseQuaternion.QuaternionToInverseRotation())
                        .Multiply(inverseCalibration)
                        .Multiply(dPixels);
        }

        public static DenseMatrix QuaternionToInverseRotation(this DenseVector quaternion)
        {
            //http://szeliski.org/Book/: page 44

            //ToDo: Do this without using inverse()
            
            
            DenseMatrix m = DenseMatrix.Identity(4);
            var x = quaternion[0];
            var y = quaternion[1];
            var z = quaternion[2];
            var w = quaternion[3];

            m[0, 0] = 1.0 - 2.0 * (y * y + z * z);
            m[0, 1] = 2.0 * (x * y - z * w);
            m[0, 2] = 2.0 * (x * z + y * w);
            m[1, 0] = 2.0 * (x * y + z * w);
            m[1, 1] = 1.0 - 2.0 * (x * x + z * z);
            m[1, 2] = 2.0 * (y * z - x * w);
            m[2, 0] = 2.0 * (x * z - y * w);
            m[2, 1] = 2.0 * (y * z + x * w);
            m[2, 2] = 1.0 - 2.0 * (x * x + y * y);

            return (DenseMatrix)m.Inverse();
        }

        public static DenseMatrix PositionToInverseTranslation(this DenseVector position)
        {
            DenseMatrix invTrans = DenseMatrix.Identity(4);
            invTrans[0, 3] = - position[0];
            invTrans[1, 3] = - position[1];
            invTrans[2, 3] = - position[2];
            return invTrans;
        }

        public static DenseMatrix CreateCalibrationMatrix(double focalLength, double centerX, double centerY)
        {
            DenseMatrix cal = DenseMatrix.Identity(4);
            cal[0, 0] = focalLength;
            cal[0, 2] = centerX;
            cal[1, 1] = focalLength;
            cal[1, 2] = centerY;
            return cal;
        }
    }
}
