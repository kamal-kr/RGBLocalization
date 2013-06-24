using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;


namespace RGBLocalization.Utility
{
    public static class MyExtensions
    {
        public static IEnumerable<T> Sample<T>(this IEnumerable<T> ie, double fraction, Random r)
        {
            return ie.Where(e => r.NextDouble() < fraction);
        }

        public static T[] ReservoirSample<T>(this IEnumerable<T> ie, int numSamples, Random r)
        {
            T[] selected = ie.Take(numSamples).ToArray();

            int n = numSamples;
            foreach (var e in ie.Skip(numSamples))
            {
                int rand = r.Next(++n);
                if (rand < numSamples)
                {
                    selected[rand] = e;
                }
            }
            return selected;
        }

        public static IEnumerable<TOut> InfiniteSelectFromWhole<TIn, TOut>(this TIn obj, Func<TIn, TOut> mapFunction)
        {
            return InfiniteEnumerate(() => mapFunction(obj));
        }

        public static IEnumerable<TOut> InfiniteEnumerate<TOut>(Func<TOut> f)
        {
            while (true)
            {
                yield return f();
            }
        }

        public static IEnumerable<T> Replicate<T>(this IEnumerable<T> ie, int n)
        {
            return Enumerable.Range(1, n).SelectMany(i => ie);
        }

        public static IEnumerable<T> ShowProgress<T>(this IEnumerable<T> ie, string label, int interval)
        {
            return ie.Select((e, i) =>
            {
                if (i % interval == 0) { Console.WriteLine("{0}: {1}", label, i); }
               return e; 
            });
        }

        public static double distanceTo(this System.Drawing.PointF p1, System.Drawing.PointF p2)
        {
            return Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
        }
    }
}
