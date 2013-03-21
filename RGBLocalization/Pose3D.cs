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
                        posePosition.PositionToTranslation()
                        .Multiply(poseQuaternion.QuaternionToRotation())
                        .Multiply(inverseCalibration)
                        .Multiply(dPixels);
        }

        public static DenseMatrix QuaternionToRotation(this DenseVector quaternion)
        {
            //http://szeliski.org/Book/: page 44
            
            DenseMatrix m = DenseMatrix.Identity(4);
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

        public static DenseMatrix PositionToTranslation(this DenseVector position)
        {
            DenseMatrix trans = DenseMatrix.Identity(4);
            trans[0, 3] = position[0];
            trans[1, 3] = position[1];
            trans[2, 3] = position[2];
            return trans;
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
