using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RGBLocalization.Utility;

namespace RGBLocalization
{
    public static partial class Pose3D
    {
        public class GradientDescentOptions<ParameterType>
        {
            public double learningRate { get; set; }
            public int numIterations { get; set; }
            public double errorTolerance { get; set; }
            public Action<int, DenseMatrix, ParameterType> iterationListener { get; set; }

            public GradientDescentOptions()
            {
                learningRate = 0.01;
                numIterations = 100;
                errorTolerance = 0;
                iterationListener = (i, err, p) => { Console.WriteLine("GD: Error_{0}: {1}", i, err.L2Norm()); };
            }
        }

        public static DenseMatrix InferRTFromScaleInvariantWorld(DenseMatrix known3DWorldPoints, 
                                                                DenseMatrix scaleInvariantWorldPoints,
                                                                GradientDescentOptions<Tuple<DenseMatrix, DenseMatrix>> gdOptions)
        {
            Debug.Assert(known3DWorldPoints.ColumnCount == scaleInvariantWorldPoints.ColumnCount);
            
            //Infer the [R|T] matrix that transforms scaleInvariantWorldPoints to known3DWorldPoints
            //to do this, we need to iteratively update our estimates for R|T and the unknown scale of each point in scaleInvariantWorldPoints
            
            //Note: The current implementation does gradient descent in batch mode and is bestsuited for small number of points (as I intend to use it)
            //To try: EM optimization for the scenario where we have a very good prior of R|T

            //Initialize R|T and scales - rather than setting them to random values, let's try identity
            DenseMatrix inferredRT = DenseMatrix.CreateRandom(4, 4, new MathNet.Numerics.Distributions.ContinuousUniform());//DenseMatrix.Identity(4);
            DenseMatrix inferredScale = DenseMatrix.CreateRandom(1, known3DWorldPoints.ColumnCount, new MathNet.Numerics.Distributions.ContinuousUniform()); //new DenseMatrix(1, known3DWorldPoints.ColumnCount, 1.0);
            

            DenseVector rowOfOnes = new DenseVector(known3DWorldPoints.ColumnCount, 1.0);
            scaleInvariantWorldPoints = (DenseMatrix)scaleInvariantWorldPoints.InsertRow(2, rowOfOnes);
            known3DWorldPoints = (DenseMatrix)known3DWorldPoints.InsertRow(3, rowOfOnes);

            Func<DenseMatrix, DenseMatrix> ColumnMagnitude = mat => mat.PointwiseMultiply(mat).SumRows().ApplyFunction(Math.Sqrt);
                

            double error = Double.MaxValue;

            for (int i = 0; i < gdOptions.numIterations && error > gdOptions.errorTolerance; i++)
            {
                DenseMatrix worldMinusTrans =
                    (DenseMatrix)
                    known3DWorldPoints.SubMatrix(0, 3, 0, known3DWorldPoints.ColumnCount)
                    .Subtract(((DenseMatrix)inferredRT.SubMatrix(0, 3, 3, 1)).Replicate(1, known3DWorldPoints.ColumnCount));

                //closed form for scale inference
                inferredScale = (DenseMatrix)ColumnMagnitude(worldMinusTrans).PointwiseDivide(ColumnMagnitude(scaleInvariantWorldPoints));
                


                //Compute the transform using the current estimates
                //Choosing clarity over performance in this part. Shall optimize as required.

                DenseMatrix infScaledWorld = (DenseMatrix)inferredScale.Replicate(3, 1)
                                               .PointwiseMultiply(scaleInvariantWorldPoints)
                                               .InsertRow(3, rowOfOnes);

                var impliedWorldPoints = inferredRT.Multiply(infScaledWorld);

                //Compute and report the error in this iteration
                DenseMatrix errorMatrix = (DenseMatrix)known3DWorldPoints.Subtract(impliedWorldPoints);
                error = errorMatrix.PointwiseMultiply(errorMatrix).SumRows().Transpose().SumRows().ApplyFunction(Math.Sqrt)[0, 0];
                gdOptions.iterationListener(i, errorMatrix, new Tuple<DenseMatrix,DenseMatrix>(inferredRT, inferredScale));
                
                //Compute the gradients
                var NegRTGradients = errorMatrix.Multiply(infScaledWorld.Transpose());
                //var NegScaleGradients = inferredRT.SubMatrix(0, 3, 0, 3)
                //                        .Multiply(scaleInvariantWorldPoints)
                //                        .PointwiseMultiply(errorMatrix.SubMatrix(0, 3, 0, known3DWorldPoints.ColumnCount))
                //                        .SumRows();

                Console.WriteLine("RT gradients:");
                Console.WriteLine("{0:0.000}", NegRTGradients);



                //Update the estimates
                inferredRT = (DenseMatrix)inferredRT.Add(NegRTGradients.Multiply(gdOptions.learningRate));
                //inferredScale = (DenseMatrix)inferredScale.Add(NegScaleGradients.Multiply(gdOptions.learningRate));

                //closed form update of RT

                Console.WriteLine("X.XT");
                Console.WriteLine(infScaledWorld.Multiply(infScaledWorld.Transpose()));
                Console.WriteLine("Det");
                Console.WriteLine(infScaledWorld.Multiply(infScaledWorld.Transpose()).Determinant());

                inferredRT = (DenseMatrix)SimpleRansac.ComputeLeastSquaresSolution(infScaledWorld.Transpose(), known3DWorldPoints.Transpose()).Transpose();
                

            }
            return inferredRT;
        }
    }
}
