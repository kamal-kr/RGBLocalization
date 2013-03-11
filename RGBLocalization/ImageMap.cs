using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RGBLocalization
{
    public class ImageMap
    {
        public static IEnumerable<T> ParsePoseData<T>(IEnumerable<string> poseLines, Func<string, string, string, string, string, string, string, string, T> map)
        {
            return
                poseLines
                .Select(l => l.Split(new char[] { ' ', ':' }, StringSplitOptions.RemoveEmptyEntries))
                .Select(l => map(l[0] + "-" + l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]));
        }

        /*public static IEnumerable<T> GetImageMap<T>(string dirName, string poseTextFile, Func<T> map)
        {
            Directory.GetFiles(dirName, "")
                
        }*/
    }
}
