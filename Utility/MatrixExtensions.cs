﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using MathNet.Numerics.LinearAlgebra.Double;


namespace RGBLocalization.Utility
{
    public static class MatrixExtensions
    {
        public static IEnumerable<Emgu.CV.Matrix<T>> EnumerableRows<T>(this Emgu.CV.Matrix<T> mat) where T : new()
        {
            return Enumerable.Range(0, mat.Height).Select(r => mat.GetRow(r));
        }

        public static IEnumerable<Emgu.CV.Matrix<T>> EnumerableCols<T>(this Emgu.CV.Matrix<T> mat) where T : new()
        {
            return Enumerable.Range(0, mat.Width).Select(r => mat.GetCol(r));
        }

        public static IEnumerable<T> EnumerateRowwise<T>(this Emgu.CV.Matrix<T> mat) where T : new()
        {
            return mat.EnumerableRows().SelectMany(m => m.EnumerableCols().Select(mc => mc[0, 0]));
        }

        public static DenseMatrix Replicate(this MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> mat, int nRows, int nCols)
        {
            return new DenseMatrix(mat.RowCount * nRows, mat.ColumnCount * nCols,
                                mat.ColumnEnumerator()
                                .SelectMany(c => c.Item2.ToArray().Replicate(nRows))
                                .Replicate(nCols)
                                .ToArray());
        }

        public static DenseMatrix SumRows(this MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> mat)
        {
            return
                (DenseMatrix)
            mat.RowEnumerator()
                .Aggregate(new DenseMatrix(1, mat.ColumnCount, 0),
                            (a, i) => (DenseMatrix)a.Add(i.Item2.ToRowMatrix()));
        }

        public static DenseMatrix ApplyFunction(this MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> mat, Func<double, double> foo)
        {
            return new DenseMatrix(mat.RowCount, mat.ColumnCount, mat.ToColumnWiseArray().Select(d => foo(d)).ToArray());
        }

        public static double Magnitude(this MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> mat)
        {
            return mat.ApplyFunction(r => r*r).SumRows().Transpose().SumRows()[0, 0];
        }

        public static string ToString(this Emgu.CV.Matrix<double> mat)
        {
            return 
            mat.EnumerableRows()
                .Aggregate(new StringBuilder(),
                            (a, i) =>
                            {
                                i.EnumerableCols().Aggregate(a, (aa, ii) => { aa.AppendFormat("{0:0.00} ", ii[0, 0]); return aa; })
                                    .AppendLine();
                                return a;
                            })
            .ToString();
        }

        public static DenseMatrix ToMatrix(this IEnumerable<System.Drawing.PointF> features, Func<System.Drawing.PointF, double[]> mapPointToRow, int numCols)
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

        public static Emgu.CV.Matrix<double> ToEmguMatrix(this MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> mat)
        {
            return new Emgu.CV.Matrix<double>(mat.ToArray());
        }

        public static DenseMatrix ToDenseMatrix(this Emgu.CV.Matrix<double> mat)
        {
            return new DenseMatrix(mat.Data);
        }
    }
}
