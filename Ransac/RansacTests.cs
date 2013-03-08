using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;
using RGBLocalization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ransac
{
    class RansacTests
    {
        
        static void Main(string[] args)
        {
            TestSimpleRansac(@"C:\Kamal\RSE\TestResults\RansacImpl");
            Console.WriteLine("Done! Hit enter!");
            Console.ReadLine();
        }

        static void TestSimpleRansac(string resultsDir)
        {
            File.WriteAllLines(Path.Combine(resultsDir, "SimpleRansac_Synthetic_FeatureNoiseSensitivity.txt"),
                Enumerable.Repeat("FeatureNoiseStdDev\tRansacError", 1).Concat(
            Enumerable.Range(0, 1000)
                .Select(i => 20.0*i/1000.0)
                .AsParallel()
                .Select(pixelNoise => new {
                                            pixelNoise,
                                            ransacError = 
                                                            SimpleRansac.RansacMatch(
                                                                SyntheticFeatureAlignmentData(15, 30, new Normal(0, pixelNoise), new ContinuousUniform(0, 700)).ToList(),
                                                                400, //the expected number of trials is 27 for the above setting
                                                                3,
                                                                10,
                                                                20,
                                                                new Random(1))
                                             })
                .OrderBy(dp => dp.pixelNoise)
                .Select(d => String.Format("{0}\t{1}", d.pixelNoise, d.ransacError))));

            File.WriteAllLines(Path.Combine(resultsDir, "SimpleRansac_Synthetic_AlignmentSensitivity.txt"),
                Enumerable.Repeat("NumGoodAligned\tNumBadAligned", 1).Concat(
            Enumerable.Range(1, 100)
                .AsParallel()
                .Select(numBadAligned => new
                {
                    numGoodAligned = 20,
                    numBadAligned,
                    ransacError =
                                    SimpleRansac.RansacMatch(
                                        SyntheticFeatureAlignmentData(20, numBadAligned, new Normal(0, 2.0), new ContinuousUniform(0, 700)).ToList(),
                                        4000, 
                                        3,
                                        8,
                                        5.0,
                                        new Random(1))
                })
                .OrderBy(dp => dp.numBadAligned)
                .Select(d => String.Format("{0}\t{1}\t{2}", d.numGoodAligned, d.numBadAligned, d.ransacError))));        
        }

        public static Func<int, IContinuousDistribution, Matrix<double>> randSourceMatrix = (n, dist) => DenseMatrix.CreateRandom(n, 2, dist).InsertColumn(2, new DenseVector(Enumerable.Repeat(1.0, n).ToArray()));

        public static Func<Tuple<int, Vector<double>>, PointF> rowToFeature = r => new PointF((float)r.Item2[0], (float)r.Item2[1]);

        public static 
            Func<IEnumerable<Tuple<int, Vector<double>>>, IEnumerable<Tuple<int, Vector<double>>>, IEnumerable<Tuple<PointF, PointF>>> alignedPointsFromRows =
            (src, tar) =>
                            src.Select(rowToFeature)
                                               .Zip(tar.Select(rowToFeature),
                                                   (o, i) => new Tuple<PointF, PointF>(o, i));

        public static Func<Matrix<double>, Matrix<double>, IContinuousDistribution, Matrix<double>> NoisyTransform = (src, trans, noiseDist) => src.Multiply(trans).Add(DenseMatrix.CreateRandom(src.RowCount, 2, noiseDist));

        public static IEnumerable<Tuple<PointF, PointF>> SyntheticFeatureAlignmentData(
            int numGoodAlignments,
            int numBadAlignments,
            IContinuousDistribution featureNoise,
            IContinuousDistribution pointDistribution)
        {
            var rigidTrans = DenseMatrix.CreateRandom(3, 2, pointDistribution);

            var sourcePoints = randSourceMatrix(numGoodAlignments, pointDistribution);
            var targetPoints = NoisyTransform(sourcePoints, rigidTrans, featureNoise);

            return alignedPointsFromRows(sourcePoints.RowEnumerator(), targetPoints.RowEnumerator())
                     .Concat(alignedPointsFromRows(randSourceMatrix(numBadAlignments, pointDistribution).RowEnumerator(), DenseMatrix.CreateRandom(numBadAlignments, 2, pointDistribution).RowEnumerator()));
        }
    }
}
