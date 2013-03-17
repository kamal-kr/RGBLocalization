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
            string workingDir = @"C:\Kamal\RSE\WorkingDirs\Visualizaton";

            var testPairs = Directory.GetFiles(dirName, "*-image.png")
                            .Select(f => new Tuple<string, string>(f, f.Replace("image", "depth")))
                            //.Take(1)
                            ;

            var imageMap =
                ImageMap.CreateImageMap(testPairs,
                                    poseFile,
                                    new RGBMatch.FeatureExtractionOptions { numPoints = 100, threshold = 30 },
                                    Pose3D.CreateCalibrationMatrix(525, 320, 240),
                                    (frameId, worldPoints) => worldPoints)
                                    .SelectMany(w => w.ColumnEnumerator())
                                    .Select(c => String.Format("{0},{1},{2}", c.Item2[0] / c.Item2[3], c.Item2[1] / c.Item2[3], c.Item2[2] / c.Item2[3]));
            
            File.WriteAllLines(Path.Combine(workingDir, "10.asc"), imageMap);
            //Console.WriteLine(imageMap.Count());
        }
    }
}