using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Features2D;
using System.IO;
using System.Drawing;
using Emgu.CV.Structure;
using RGBLocalization.Utility;
using Emgu.CV.Util;
using Emgu.CV;
using MathNet.Numerics.LinearAlgebra.Double;

namespace RGBLocalization
{
    public class StreamTracker
    {
        public static void MatchInMap(string rgbImage, string mapFile)
        {
            //var imageMap = ImageMap.LoadImageMap(mapFile, (frameId, worldPoint, featureDesc) => new { frameId, worldPoint, featureDesc }).ToArray();
            
            //var featurePoints2D = new ImageFeatureExtraction
            //                            .FeatureExtractionOptions()
            //                            .DoExtract(new Emgu.CV.Image<Gray, byte>(rgbImage))
            //                            .ToArray();



        }
    }
}
