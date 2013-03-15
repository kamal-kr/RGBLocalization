using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RGBLocalization
{
    public class ImageMapTests
    {
        public static void TestImageMap()
        {
            string dirName = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump";
            string poseFile = @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt";

            var testPairs = Directory.GetFiles(dirName, "*-image.png")
                            .Select(f => new Tuple<string, string>(f, f.Replace("image", "depth")))
                //.Take(10);
                            ;

            Console.WriteLine(testPairs.Count());
            /*var imageMap = 
                ImageMap.CreateImageMap(testPairs,
                                    poseFile,
                                    new RGBMatch.FeatureExtractionOptions { numPoints = 100, threshold = 100 },
                                    Pose3D.CreateCalibrationMatrix(525, 320, 240),
                                    (frameId, worldPoints) => new { frameId, worldPoints })
                    .ToDictionary(k => k.frameId, k => k.worldPoints);
            */
            //Console.WriteLine(imageMap.Count());
        }
    }
}