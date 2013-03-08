using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RGBLocalization.Utility;
using System.IO;
using Emgu.CV.Features2D;


namespace RGBLocalization
{
    class RGBMatchTests
    {
        public static void DebugRGBMatch()
        {
            //string queryImage = @"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389259-225498729-image.png";
            string queryImage = @"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389276-824813931-image.png";
            //string mapImage = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\1302389028-661940549-image.png";
            //string mapImage = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\1302389031-393932434-image.png";
            string mapImage = @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\1302389105-603398256-image.png";
            string debugDir = @"C:\Kamal\RSE\WorkingDirs\RGBMatchTest\";


            RGBMatch.DoRGBMatch(
                    new Emgu.CV.Image<Gray, byte>(queryImage),
                    new Emgu.CV.Image<Gray, byte>(mapImage),
                    RGBMatch.FastFeatureExt,
                    10,
                    1000,
                    RGBMatch.BriefFeatureDescriptorFunc,
                    RGBMatch.HammingDist,
                    25,
                    2,
                    2000,
                    new Random(1),
                    fp => RGBMatch.VisualizeFeaturePairs(fp, queryImage, mapImage, 
                            Path.Combine(debugDir, "Neighbors_" + Path.GetFileNameWithoutExtension(queryImage) + "_" + Path.GetFileNameWithoutExtension(mapImage) + ".png")),
                    fp => RGBMatch.VisualizeFeaturePairs(fp, queryImage, mapImage, 
                            Path.Combine(debugDir, "OptimalPairs_" + Path.GetFileNameWithoutExtension(queryImage) + "_" + Path.GetFileNameWithoutExtension(mapImage) + ".png"))
                    );
        }
        
        public static void CmdLineImageSearch(IEnumerable<string> imageMap)
        {
            /*
             * File.WriteAllLines(@"C:\Kamal\RSE\TestResults\FastFeatures\FF_histogramOfNumFeatures.txt",
            imageMap
            .AsParallel()
            .Select(im => RGBMatch.FastFeatureExt(new Emgu.CV.Image<Gray, byte>(im), 10, int.MaxValue))
            .Select(feat => feat == null ? -1 : feat.Count())
            .GroupBy(f => f)
            .Select(f => new { f.Key, numImages = f.Count() })
            .OrderBy(f => f.Key)
            .Select(f => String.Format("{0}\t{1}", f.Key, f.numImages)));
            */


            imageMap = new string[] { @"C:\Kamal\RSE\RawData\espresso-1.bag.dump\espresso-1.bag.dump\espresso-1.bag.dump\1302389028-661940549-image.png" };
            //imageMap = new string[] { @"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389259-225498729-image.png" };
            //Console.WriteLine(imageMap.Count());
            //foreach (var query in MyExtensions.InfiniteEnumerate(Console.ReadLine))
            foreach(var query in new string[] {@"C:\Kamal\RSE\RawData\espresso-2.bag.dump\espresso-2.bag\espresso-2.bag.dump\1302389259-225498729-image.png"})
            {
                foreach (var res in FetchMatchingImages(query, imageMap))
                {
                    Console.WriteLine("{0}\t{1}", res.Item1, res.Item2);
                    //Console.WriteLine(FetchMatchingImages(query, ).Count());
                }
            }
        }

       static IEnumerable<Tuple<string, double>> FetchMatchingImages(string queryImage, IEnumerable<string> imageMap)
        {
           var q = new Emgu.CV.Image<Gray, byte>(queryImage);
           return
               imageMap
               //.AsParallel()
               .Select(im => new
                                   {
                                       imageFile = im,
                                       matchError =
                                           RGBMatch.DoRGBMatch(q,
                                               new Emgu.CV.Image<Gray, byte>(im),
                                               RGBMatch.FastFeatureExt,
                                               10,
                                               100,
                                               RGBMatch.BriefFeatureDescriptorFunc,
                                               RGBMatch.HammingDist,
                                               20,
                                               2,
                                               2,
                                               new Random(1),
                                               fp => 1,//RGBMatch.VisualizeFeaturePairs(fp, queryImage, im, 
                                                       // @"C:\Kamal\RSE\WorkingDirs\RGBMatchTest\" + Path.GetFileNameWithoutExtension(queryImage) + "_" + Path.GetFileNameWithoutExtension(im) + ".png"),
                                               fp => 1
                                               )
                                   })
               .ShowProgress("Attempted" , 1)
               //.Where(res => res.matchError < 10)
               .ShowProgress("Matched", 1)
               .OrderBy(res => res.matchError)
               .Select(r => new Tuple<string, double>(r.imageFile, r.matchError));
        }
    }
}
