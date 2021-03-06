﻿using System;
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
        public class RansacOptions
        {
            public RansacOptions()
            {
                numTrials = 100;
                numMinSamples = 3;
                minNumInliers = 10;
                sqInlierErrorThreshold = 0.01;
                rand = new Random(0);
            }

            public int numTrials {get;set;}
            public int numMinSamples { get; set; }
            public int minNumInliers { get; set; }
            public double sqInlierErrorThreshold { get; set; }
            public Random rand { get; set; }
        }


        public static Tuple<Md, double> Ransac<Dp, Md>(
                                                    IEnumerable<Dp> dataPoints, 
                                                    Func<Dp[], Tuple<Md, double>> modelFitter, 
                                                    Func<Md, Dp, double> modelEvaluator,
                                                    RansacOptions options)
            where Md: class
        {
            var samples = 
            dataPoints
                .InfiniteSelectFromWhole(ie => ie.ReservoirSample(options.numMinSamples, options.rand))
                .Take(options.numTrials); //take many samples

            var models = samples
                            .Select(modelFitter) //model each one of them
                            .Where(me => me.Item2 < options.sqInlierErrorThreshold) //keep only those where the error of modelling the minsamples is small
                            .Select(me => me.Item1); 

            Func<Md, Dp[]> GetInliers = m => dataPoints.Where(d => modelEvaluator(m, d) < options.sqInlierErrorThreshold).ToArray();

            return
            models
                .Select(GetInliers)
                .Where(inl => inl.Count() >= options.minNumInliers) //retain only those will the required number of inliers
                .Select(modelFitter)//recompute models and errors by fitting all inliers
                .Aggregate(new Tuple<Md, double>(null, Double.MaxValue),
                            (a, i) => a.Item2 > i.Item2? i : a); //find the one with the least error
        }

        public static Tuple<double, List<Tuple<PointF, PointF>>> RansacMatch(
            List<Tuple<PointF, PointF>> featurePairs, 
            RansacOptions options,
            Random rand)
        {
            if (featurePairs.Count() <= options.minNumInliers)
            {
                return new Tuple<double,List<Tuple<PointF,PointF>>>(Double.MaxValue, null);
            }

            var fullSourceMat = featurePairs.Select(fp => fp.Item1).ToMatrix(pt => new double[] { pt.X, pt.Y, 1 });
            var fullDestMat = featurePairs.Select(fp => fp.Item2).ToMatrix(pt => new double[] { pt.X, pt.Y});
            
            return 
            featurePairs
            .InfiniteSelectFromWhole(fpSet => fpSet.ReservoirSample(options.numMinSamples, rand))
            .Take(options.numTrials)
            .AsParallel()
            .Select(samp =>
                            {
                                DenseMatrix designMatrix = samp.Select(s => s.Item1).ToMatrix(f => new double[] { f.X, f.Y, 1 });
                                DenseMatrix lseSolution = ComputeLeastSquaresSolution(designMatrix, samp.Select(s => s.Item2).ToMatrix(f => new double[] { f.X, f.Y }));
                                
                                return fullSourceMat.Multiply(lseSolution).Subtract(fullDestMat)
                                                .RowEnumerator()
                                                .Select(ri => new { rowID = ri.Item1, rowErr = ri.Item2.Select(e => e * e).Sum() })
                                                .Where(ri => ri.rowErr < options.sqInlierErrorThreshold)
                                                .Select(ri => featurePairs[ri.rowID])
                                                .ToList();
                            })
            .Where(inliers => inliers.Count() > options.minNumInliers)
            .Select(inliers => 
                            {   //learn the rigid transformation again by using all the inliers
                                var designMatrix = inliers.Select(s => s.Item1).ToMatrix(f => new double[] { f.X, f.Y, 1 });
                                var targetMatrix = inliers.Select(s => s.Item2).ToMatrix(f => new double[] { f.X, f.Y });
                                var lseSolution = ComputeLeastSquaresSolution(designMatrix, targetMatrix);

                                return new Tuple<double, List<Tuple<PointF, PointF>>>(
                                        designMatrix.Multiply(lseSolution).Subtract(targetMatrix).L2Norm() / inliers.Count(),
                                        inliers);

                            })
            .Aggregate(new Tuple<double, List<Tuple<PointF, PointF>>>(Double.MaxValue, null), 
            (a, i) => i.Item1 < a.Item1? i:a);
        }

    public static DenseMatrix ComputeLeastSquaresSolution(MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> X, MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> Y)
    {
        var XTranspose = X.Transpose();
        return (DenseMatrix)XTranspose.Multiply(X).Inverse().Multiply(XTranspose).Multiply(Y);
    }

    }
}
