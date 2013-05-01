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
            string streamDirName = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump";
            string poseFile = @"C:\Kamal\RSE\RawData\espresso-1-fs-0\espresso-1-fs-0\espresso-1-fs-0\loop_closure\loop-closure.txt";
            string workingDir = @"C:\Kamal\RSE\WorkingDirs\Visualizaton";

            //1302389028-661940549
            var testPairs = Directory.GetFiles(streamDirName, "*-image.png")
                            .Select(f => new Tuple<string, string>(f, f.Replace("image", "depth")));
            
            ImageMap.SaveImageMap(testPairs, poseFile, Path.Combine(workingDir, Path.GetFileName(streamDirName)+".map"));
        }
    }
}