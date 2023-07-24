using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace NeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var network = new Network(new int[] { 784, 50, 25, 10 });
            CsvToArray csvLoader = new CsvToArray();
            double[][] trainInput;
            double[][] trainOutput;
            csvLoader.LoadCsvToArray(args[0], 1, out trainInput, out trainOutput);
            double[][] testInput;
            double[][] testOutput;
            //Convert expected output to array.
            for (int i = 0; i < trainInput.Length; i++)
            {
                double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                temp[(int)trainInput[i][0]] = 1;
            }
            csvLoader.LoadCsvToArray(args[1], 1, out testInput, out testOutput);
            for (int i = 0; i < testInput.Length; i++)
            {
                double[] temp = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                temp[(int)testInput[i][0]] = 1;
            }
            network.Train(3, 0.05d, trainInput, trainOutput, testInput, testOutput);
        }
    }

    public class CsvToArray
    {
        public void LoadCsvToArray(string filePath, int A, out double[][] arrayA, out double[][] arrayB)
        {
            List<double[]> tempArrayA = new List<double[]>();
            List<double[]> tempArrayB = new List<double[]>();

            using (var reader = new StreamReader(filePath))
            {
                string line;
                int lineCount = 0;

                while ((line = reader.ReadLine()) != null)
                {
                    if (lineCount == 0)
                    {
                        lineCount++;
                        continue; // Skip the first row
                    }

                    string[] values = line.Split(',');

                    if (values.Length < A)
                    {
                        Console.WriteLine($"Warning: Line {lineCount + 1} contains less than {A} elements.");
                        continue;
                    }

                    double[] rowArrayA = new double[A];
                    double[] rowArrayB = new double[values.Length - A];

                    for (int i = 0; i < values.Length; i++)
                    {
                        double nodeValue;

                        if (!double.TryParse(values[i], NumberStyles.Float, CultureInfo.InvariantCulture, out nodeValue))
                        {
                            Console.WriteLine($"Error: Line {lineCount}, Element {i} is not a valid double value.");
                            continue;
                        }

                        if (i < A)
                            rowArrayA[i] = nodeValue;
                        else
                            rowArrayB[i - A] = nodeValue;
                    }

                    tempArrayA.Add(rowArrayA);
                    tempArrayB.Add(rowArrayB);

                    lineCount++;
                }
            }

            arrayA = tempArrayA.ToArray();
            arrayB = tempArrayB.ToArray();
        }
    }
}

