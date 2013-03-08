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

        public static T[] ReservoirSample<T>(this List<T> ie, int numSamples, Random r)
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

        public static IEnumerable<Emgu.CV.Matrix<T>> EnumerableRows<T>(this Emgu.CV.Matrix<T> mat) where T:new()
        {
            return Enumerable.Range(0, mat.Height).Select(r => mat.GetRow(r));
        }

        public static IEnumerable<Emgu.CV.Matrix<T>> EnumerableCols<T>(this Emgu.CV.Matrix<T> mat) where T : new()
        {
            return Enumerable.Range(0, mat.Width).Select(r => mat.GetCol(r));
        }

        public static IEnumerable<T> EnumerateRowwise<T>(this Emgu.CV.Matrix<T> mat) where T:new()
        {
            return mat.EnumerableRows().SelectMany(m => m.EnumerableCols().Select(mc => mc[0, 0]));
        }

        public static IEnumerable<T> ShowProgress<T>(this IEnumerable<T> ie, string label, int interval)
        {
            return ie.Select((e, i) =>
            {
                if (i % interval == 0) { Console.WriteLine("{0}: {1}", label, i); }
               return e; 
            });
        }
    }
}
