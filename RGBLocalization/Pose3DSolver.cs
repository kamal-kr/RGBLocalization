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
            public int errorTolerance { get; set; }
            public Action<int, double, ParameterType> iterationListener { get; set; }

            public GradientDescentOptions()
            {
                learningRate = 0.01;
                numIterations = 100;
                errorTolerance = 0;
                iterationListener = (i, err, p) => { Console.WriteLine("GD: Error_{0}: {1}", i, err); };
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="known3DWorldPoints"></param>
        /// <param name="scaleInvariantWorldPoints"></param>
        /// <param name="inferredScale"></param>
        /// <returns></returns>
        public static DenseMatrix InferRTFromScaleInvariantWorld(DenseMatrix known3DWorldPoints, 
                                                                DenseMatrix scaleInvariantWorldPoints,
                                                                GradientDescentOptions<Tuple<DenseMatrix, DenseMatrix>> gdOptions,
                                                                out DenseMatrix inferredScale)
        {
            Debug.Assert(known3DWorldPoints.ColumnCount == scaleInvariantWorldPoints.ColumnCount);
            
            //Infer the [R|T] matrix that transforms scaleInvariantWorldPoints to known3DWorldPoints
            //to do this, we need to iteratively update our estimates for R|T and the unknown scale of each point in scaleInvariantWorldPoints
            
            //Note: The current implementation does gradient descent in batch mode and is bestsuited for small number of points (as I intend to use it)
            //To try: EM optimization for the scenario where we have a very good prior of R|T

            //Initialize R|T and scales - rather than setting them to random values, let's try identity
            DenseMatrix inferredRT = DenseMatrix.Identity(4);
            inferredScale = new DenseMatrix(1, known3DWorldPoints.ColumnCount, 1.0);
            
            DenseVector rowOfOnes = new DenseVector(known3DWorldPoints.ColumnCount, 1.0);
            scaleInvariantWorldPoints = (DenseMatrix)scaleInvariantWorldPoints.InsertRow(2, rowOfOnes);
            known3DWorldPoints = (DenseMatrix)known3DWorldPoints.InsertRow(3, rowOfOnes);


            double error = Double.MaxValue;

            for (int i = 0; i < gdOptions.numIterations && error > gdOptions.errorTolerance; i++)
            {
                //Compute the transform using the current estimates
                //Choosing clarity over performance in this part. Shall optimize as required.

                DenseMatrix infScaledWorld = (DenseMatrix)inferredScale.Replicate(3, 1)
                                               .PointwiseMultiply(scaleInvariantWorldPoints)
                                               .InsertRow(3, rowOfOnes);

                var impliedWorldPoints = inferredRT.Multiply(infScaledWorld);

                //Compute and report the error in this iteration
                var errorMatrix = known3DWorldPoints.Subtract(impliedWorldPoints);
                
                gdOptions.iterationListener(i, errorMatrix.L2Norm(), new Tuple<DenseMatrix,DenseMatrix>(inferredRT, inferredScale));
                
                //Compute the gradients
                var NegRTGradients = errorMatrix.Multiply(infScaledWorld.Transpose());
                var NegScaleGradients = inferredRT.SubMatrix(0, 3, 0, 3)
                                            .Multiply(scaleInvariantWorldPoints)
                                            .SumRows();


                //Update the estimates
                inferredRT.Add(NegRTGradients.Multiply(gdOptions.learningRate));
                inferredScale.Add(NegScaleGradients.Multiply(gdOptions.learningRate));
            }
            return null;
        }
    }
}
