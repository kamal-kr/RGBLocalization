using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RGBLocalization.Utility;
using MathNet.Numerics.LinearAlgebra.Double;

namespace RGBLocalization
{
    public class SimpleRansac
    {

        public static Tuple<double, List<Tuple<PointF, PointF>>> RansacMatch(
            List<Tuple<PointF, PointF>> featurePairs, 
            int numTrials,
            int numMinSamples,
            int minNumInliers,
            double sqInlierErrorThreshold,
            Random rand)
        {
            if (featurePairs.Count() <= minNumInliers)
            {
                return new Tuple<double,List<Tuple<PointF,PointF>>>(Double.MaxValue, null);
            }

            var fullSourceMat = CreateMatrixFromFeatures(featurePairs.Select(fp => fp.Item1), pt => new double[] { pt.X, pt.Y, 1 }, 3);
            var fullDestMat = CreateMatrixFromFeatures(featurePairs.Select(fp => fp.Item2), pt => new double[] { pt.X, pt.Y}, 2);
            
            return 
            featurePairs
            .InfiniteSelectFromWhole(fpSet => fpSet.ReservoirSample(numMinSamples, rand))
            .Take(numTrials)
            .AsParallel()
            .Select(samp =>
                            {
                                DenseMatrix designMatrix = CreateMatrixFromFeatures(samp.Select(s => s.Item1), f => new double[] { f.X, f.Y, 1 }, 3);
                                DenseMatrix lseSolution = ComputeLeastSquaresSolution(designMatrix, CreateMatrixFromFeatures(samp.Select(s => s.Item2), f => new double[] { f.X, f.Y }, 2));
                                
                                return fullSourceMat.Multiply(lseSolution).Subtract(fullDestMat)
                                                .RowEnumerator()
                                                .Select(ri => new { rowID = ri.Item1, rowErr = ri.Item2.Select(e => e * e).Sum() })
                                                .Where(ri => ri.rowErr < sqInlierErrorThreshold)
                                                .Select(ri => featurePairs[ri.rowID])
                                                .ToList();
                            })
            .Where(inliers => inliers.Count() > minNumInliers)
            .Select(inliers => 
                            {   //learn the rigid transformation again by using all the inliers
                                var designMatrix = CreateMatrixFromFeatures(inliers.Select(s => s.Item1), f => new double[] { f.X, f.Y, 1 }, 3);
                                var targetMatrix = CreateMatrixFromFeatures(inliers.Select(s => s.Item2), f => new double[] { f.X, f.Y }, 2);
                                var lseSolution = ComputeLeastSquaresSolution(designMatrix, targetMatrix);

                                return new Tuple<double, List<Tuple<PointF, PointF>>>(
                                        designMatrix.Multiply(lseSolution).Subtract(targetMatrix).L2Norm() / inliers.Count(),
                                        inliers);

                            })
            .Aggregate(new Tuple<double, List<Tuple<PointF, PointF>>>(Double.MaxValue, null), 
            (a, i) => i.Item1 < a.Item1? i:a);
        }

    public static DenseMatrix CreateMatrixFromFeatures(IEnumerable<PointF> features, Func<PointF, double[]> mapPointToRow, int numCols)
    {
        return 
        features
        .Select((fp, i) => new { fp, i })
        .Aggregate(
                new DenseMatrix(features.Count(), numCols),
                (a, i) =>
                {
                    a.SetRow(i.i, mapPointToRow(i.fp));
                    return a;
                }
        );
    }

    public static DenseMatrix ComputeLeastSquaresSolution(DenseMatrix X, DenseMatrix Y)
    {
        var XTranspose = X.Transpose();
        return (DenseMatrix)XTranspose.Multiply(X).Inverse().Multiply(XTranspose).Multiply(Y);
    }

    }
}
