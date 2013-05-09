using System;
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

        public static DenseMatrix Replicate(this DenseMatrix mat, int nRows, int nCols)
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
                .Aggregate(DenseMatrix.Create(1, mat.ColumnCount, (i, j) => 0),
                            (a, i) =>
                            {
                                a.Add(i.Item2.ToRowMatrix());
                                return a;
                            });
        }
    }
}
